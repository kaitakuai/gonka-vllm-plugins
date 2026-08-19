# SPDX-License-Identifier: Apache-2.0
"""PoC as a request among requests — Ilya's premise on the plugin frame.

PoC nonces enter the engine as NORMAL generate() requests (identity rides
``SamplingParams.extra_args``, flat keys ``gonka_poc_*``) and live in
continuous batching next to chat. No collective_rpc execution, no batch
lease: paged KV like chat. The two residual seams do all the coupling:

  * pre_forward_hooks  — write this step's PoC row layout into the
    address-stable PoCNativeState buffers (mask / reflections / routing /
    synthetic embeds), scattered at the rows the scheduler chose;
  * post_forward_hooks — snap per-step k from the hidden states of PoC rows
    (same math as the dedicated chunk: pick -> project -> snap -> prev_k).

EXPERIMENTAL scope (workability): TP=1, single block_hash per step,
generation only (no teacher forcing), preemption of a PoC row aborts its
nonce (prev_k does not survive recompute). The consensus question this
build exists to measure: do engine-loop trajectories match the dedicated
chunk bit-for-bit, alone and under chat load.
"""
import logging
import threading
from typing import Any, Dict, List, Optional

import torch

from gonka_poc.poc.decode_chain import next_prev_k
from gonka_poc.poc.gpu_random import (
    decode_base_seeds,
    generate_decode_inputs_gpu,
    generate_inputs,
    random_pick_indices,
    random_pick_indices_gpu,
    route_base_seed,
    _seed_from_string,
)
from gonka_poc.poc.sphere import (
    SPHERE_DIM,
    get_sphere_codebook,
    project_to_sphere,
    snap_with_margin,
)

logger = logging.getLogger(__name__)

XA_PREFIX = "gonka_poc_"          # flat extra_args keys (vllm_xargs-friendly)
_LOCK = threading.Lock()


class _Req:
    __slots__ = ("nonce", "bh", "pk", "seq_len", "max_tokens", "base_seed",
                 "prev_k", "step", "ks", "margins", "prefill_embeds",
                 "prompt_done", "aborted", "route_window")

    def __init__(self, nonce, bh, pk, seq_len, max_tokens, route_window):
        self.nonce = int(nonce); self.bh = bh; self.pk = pk
        self.seq_len = int(seq_len); self.max_tokens = int(max_tokens)
        self.route_window = int(route_window)
        self.base_seed = _seed_from_string(f"{bh}_{pk}_nonce{nonce}")
        self.prev_k: int = 0
        self.step: int = -1            # -1 = префилл ещё не снят
        self.ks: List[int] = []
        self.margins: List[float] = []
        self.prefill_embeds: Optional[torch.Tensor] = None  # [seq_len, H] lazily
        self.prompt_done = False
        self.aborted = False


class EngineFlow:
    """Per-worker singleton driving both hooks + artifact store."""

    def __init__(self):
        self.reqs: Dict[str, _Req] = {}
        self.done: Dict[str, dict] = {}
        self._plan: List[tuple] = []   # (req_id, flat_row, step) — extraction plan
        self._codebook = None
        self._hh_bh = None

    # ------------------------------------------------------------- intake --
    def _register_new(self, scheduler_output) -> None:
        for nr in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
            sp = getattr(nr, "sampling_params", None)
            xa = getattr(sp, "extra_args", None) if sp is not None else None
            if not xa or (XA_PREFIX + "nonce") not in xa:
                continue
            rid = nr.req_id
            self.reqs[rid] = _Req(
                xa[XA_PREFIX + "nonce"], xa[XA_PREFIX + "block_hash"],
                xa[XA_PREFIX + "public_key"], xa[XA_PREFIX + "seq_len"],
                xa.get(XA_PREFIX + "max_tokens", 256),
                xa.get(XA_PREFIX + "route_window", 256))
            logger.info("engine-flow: PoC request %s nonce=%s joined the loop",
                        rid, xa[XA_PREFIX + "nonce"])

    # -------------------------------------------------------- pre-forward --
    def pre_forward(self, runner, scheduler_output, input_ids, positions,
                    inputs_embeds, attn_metadata) -> None:
        self._plan = []
        if scheduler_output is None:
            return
        self._register_new(scheduler_output)
        if not self.reqs:
            return
        model = getattr(runner, "model", None)
        state = getattr(model, "_poc_native_state", None)
        if state is None:
            raise RuntimeError(
                "engine-flow: model has no _poc_native_state — PoC math is "
                "not attached (unregistered architecture?)")
        tp = getattr(getattr(getattr(runner, "vllm_config", None),
                             "parallel_config", None),
                     "tensor_parallel_size", 1)
        if tp and int(tp) > 1:
            raise RuntimeError("engine-flow experimental build is TP=1 only")

        ib = runner.input_batch
        idx_of = ib.req_id_to_index
        nsched: Dict[str, int] = dict(scheduler_output.num_scheduled_tokens)
        # плоская раскладка токенов шага: порядок персистентного батча
        order = sorted(((i, rid) for rid, i in idx_of.items()
                        if nsched.get(rid, 0) > 0))
        spans: Dict[str, tuple] = {}
        off = 0
        for _, rid in order:
            n = nsched[rid]
            spans[rid] = (off, n)
            off += n

        # финализация ушедших (finished/aborted) запросов
        for rid in list(self.reqs):
            if rid in getattr(scheduler_output, "finished_req_ids", ()) or ():
                self._finalize(rid)

        poc_rows: List[int] = []
        bh = None
        state.mask.zero_()
        dev = state.embeds.device
        dt = state.embeds.dtype
        H = state.embeds.shape[1]
        for rid, req in list(self.reqs.items()):
            if rid not in spans or req.aborted:
                continue
            if bh is None:
                bh = req.bh
            elif bh != req.bh:
                raise RuntimeError("engine-flow: single block_hash per step")
            start, n = spans[rid]
            row_idx = idx_of[rid]
            computed = int(ib.num_computed_tokens_cpu[row_idx])
            if not req.prompt_done:
                # префилл (возможно чанкованный): наш кусок = промпт-токены
                if req.prefill_embeds is None:
                    req.prefill_embeds = generate_inputs(
                        req.bh, req.pk, [req.nonce], dim=H,
                        seq_len=req.seq_len, device=dev, dtype=dt)[0]
                take = min(n, max(req.seq_len - computed, 0))
                if take <= 0:
                    continue
                sl = slice(start, start + take)
                state.embeds[sl].copy_(
                    req.prefill_embeds[computed:computed + take])
                state.mask[sl].fill_(True)
                self._route_span(state, req, sl, step=0)
                poc_rows.extend(range(start, start + take))
                if computed + take >= req.seq_len:
                    req.prompt_done = True
                    # k0 снимается с ПОСЛЕДНЕГО промпт-токена этого шага
                    self._plan.append((rid, start + take - 1, 0))
            else:
                # декод-шаг: 1 токен
                if computed < req.seq_len:
                    continue
                step = req.step + 1
                if step > req.max_tokens:
                    continue
                base_t = torch.tensor([req.base_seed], dtype=torch.int64,
                                      device=dev)
                prev_t = torch.tensor([req.prev_k], dtype=torch.int64,
                                      device=dev)
                emb = generate_decode_inputs_gpu(base_t, prev_t, step, H, dev,
                                                 dtype=dt)
                sl = slice(start, start + 1)
                state.embeds[sl].copy_(emb)
                state.mask[sl].fill_(True)
                self._route_span(state, req, sl, step=step)
                poc_rows.append(start)
                self._plan.append((rid, start, step))

        if poc_rows and bh is not None:
            self._set_reflections(state, bh)
        # прямые записи в буферы обходят кэш-ключи state — сбросить их,
        # чтобы выделенный путь после нас не переиспользовал чужую раскладку
        state._rows_key = None
        state._route_key = None

    def _set_reflections(self, state, bh) -> None:
        if self._hh_bh == bh:
            return
        vs = state._hh_vectors(bh, None, state.vectors[0].dtype)
        for li in range(state.num_layers):
            state.vectors[li].copy_(vs[li].unsqueeze(0))
        self._hh_bh = bh

    def _route_span(self, state, req, sl, step: int) -> None:
        n = sl.stop - sl.start
        for li, buf in state._route_base:
            buf[sl].fill_(_seed_from_string(
                route_base_seed(req.bh, req.nonce, li)))
        state.route_step[sl].fill_(int(step))

    # ------------------------------------------------------- post-forward --
    def post_forward(self, runner, scheduler_output, hidden_states) -> None:
        if not self._plan or not isinstance(hidden_states, torch.Tensor):
            return
        state = runner.model._poc_native_state
        dev = hidden_states.device
        if self._codebook is None:
            self._codebook = get_sphere_codebook().to(device=dev)
        for rid, row, step in self._plan:
            req = self.reqs.get(rid)
            if req is None:
                continue
            h = hidden_states[row:row + 1].float()
            if step == 0:
                sph = random_pick_indices(req.bh, req.pk, [req.nonce],
                                          h.shape[1], SPHERE_DIM, dev)
            else:
                base_t = torch.tensor([req.base_seed], dtype=torch.int64,
                                      device=dev)
                prev_t = torch.tensor([req.prev_k], dtype=torch.int64,
                                      device=dev)
                sph = random_pick_indices_gpu(base_t, prev_t, step,
                                              h.shape[1], SPHERE_DIM, dev)
            q = project_to_sphere(torch.gather(h, 1, sph))
            k, bad, margin = snap_with_margin(q, self._codebook)
            k_i = int(k.item())
            req.ks.append(k_i)
            req.margins.append(float(margin.item()))
            req.prev_k = int(next_prev_k(k, None).item())
            req.step = step
            if step >= req.max_tokens:
                self._finalize(rid)
        self._plan = []

    def _finalize(self, rid: str) -> None:
        req = self.reqs.pop(rid, None)
        if req is None:
            return
        self.done[rid] = {
            "nonce": req.nonce,
            "k_points_steps": req.ks,
            "n_steps": len(req.ks),
            "aborted": req.aborted or len(req.ks) < req.max_tokens + 1,
        }

    # ------------------------------------------------------------- drain --
    def collect(self) -> dict:
        with _LOCK:
            out = self.done
            self.done = {}
        return {"artifacts": list(out.values()),
                "in_flight": len(self.reqs)}


FLOW = EngineFlow()


def install(runner) -> bool:
    import os
    if os.environ.get("POC_ENGINE_FLOW", "0") != "1":
        return False
    pre = getattr(runner, "pre_forward_hooks", None)
    post = getattr(runner, "post_forward_hooks", None)
    if pre is None or post is None:
        raise RuntimeError(
            "engine-flow: residual lacks pre/post forward hook seams "
            "(need kaitakuai/vllm branch mixed/poc-as-request)")
    if FLOW.pre_forward not in pre:
        pre.append(FLOW.pre_forward)
    if FLOW.post_forward not in post:
        post.append(FLOW.post_forward)
    return True
