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
    (same math as the dedicated chunk: pick -> project -> snap -> prev_k),
    chain kept ON DEVICE (no per-step host syncs; async scheduling intact).

CLIENT CONTRACT (per nonce, /v1/completions):
    prompt: exactly gonka_poc_seq_len token ids;
    max_tokens = gonka_poc_max_tokens + 1  (257 sampling forwards = steps
    0..256: the prefill forward samples token #1);
    ignore_eos = true (and ideally min_tokens = max_tokens — synthetic
    hiddens sample EOS with nonzero probability);
    vllm_xargs: gonka_poc_{nonce, block_hash, public_key, seq_len,
    max_tokens, route_window}.
    Server must run WITHOUT prefix caching for engine-flow rounds.

EXPERIMENTAL scope (workability): TP=1, one block_hash served per step
(foreign-hash rows abort, not raise), generation only. Preemption/resume,
orphaned finishes, cache hits, prompt-length or route-window mismatches all
ABORT the nonce loudly rather than corrupt the chain.
"""
import logging
import os
import threading
from typing import Dict, List, Optional

import torch

from gonka_poc.poc.decode_chain import next_prev_k
from gonka_poc.poc.gpu_random import (
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
                 "base_t", "prev_k", "step", "ks", "margins", "route_seeds",
                 "prefill_embeds", "prompt_done", "aborted", "route_window")

    def __init__(self, nonce, bh, pk, seq_len, max_tokens, route_window):
        self.nonce = int(nonce); self.bh = str(bh); self.pk = str(pk)
        self.seq_len = int(seq_len); self.max_tokens = int(max_tokens)
        self.route_window = int(route_window)
        self.base_seed = _seed_from_string(f"{self.bh}_{self.pk}_nonce{self.nonce}")
        self.base_t: Optional[torch.Tensor] = None       # [1] int64, device
        self.prev_k: Optional[torch.Tensor] = None       # [1] int64, device
        self.step: int = -1            # -1 = префилл ещё не снят
        self.ks: List[torch.Tensor] = []                 # [1] int64 each, device
        self.margins: List[torch.Tensor] = []
        self.route_seeds: Optional[List[int]] = None     # per-layer, cached
        self.prefill_embeds: Optional[torch.Tensor] = None
        self.prompt_done = False
        self.aborted = False

    def dev_init(self, device):
        if self.base_t is None:
            self.base_t = torch.tensor([self.base_seed], dtype=torch.int64,
                                       device=device)
            self.prev_k = torch.zeros(1, dtype=torch.int64, device=device)


class EngineFlow:
    """Per-worker singleton driving both hooks + artifact store."""

    def __init__(self):
        self.reqs: Dict[str, _Req] = {}
        self.done: List[dict] = []
        self._plan: List[tuple] = []   # (req_id, flat_row, step)
        self._codebook = None
        self._mask_dirty = False
        self._runner = None

    # ------------------------------------------------------------- intake --
    def _register_new(self, scheduler_output) -> None:
        for nr in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
            sp = getattr(nr, "sampling_params", None)
            xa = getattr(sp, "extra_args", None) if sp is not None else None
            if not xa or (XA_PREFIX + "nonce") not in xa:
                continue
            rid = nr.req_id
            try:
                req = _Req(
                    xa[XA_PREFIX + "nonce"], xa[XA_PREFIX + "block_hash"],
                    xa[XA_PREFIX + "public_key"], xa[XA_PREFIX + "seq_len"],
                    xa.get(XA_PREFIX + "max_tokens", 256),
                    xa.get(XA_PREFIX + "route_window", 256))
            except (KeyError, ValueError, TypeError) as e:
                logger.warning("engine-flow: %s malformed gonka_poc_* (%r) — "
                               "treated as chat", rid, e)
                continue
            # sampling contract: steps 0..N need N+1 sampling forwards
            if (getattr(sp, "max_tokens", 0) or 0) < req.max_tokens + 1 \
                    or not getattr(sp, "ignore_eos", False):
                logger.error(
                    "engine-flow: %s bad sampling contract (max_tokens=%s "
                    "need>=%s, ignore_eos=%s) — skipped", rid,
                    getattr(sp, "max_tokens", None), req.max_tokens + 1,
                    getattr(sp, "ignore_eos", None))
                continue
            self.reqs[rid] = req
            logger.info("engine-flow: PoC request %s nonce=%s joined the loop",
                        rid, req.nonce)

    # -------------------------------------------------------- pre-forward --
    def pre_forward(self, runner, scheduler_output, input_ids, positions,
                    inputs_embeds, attn_metadata) -> None:
        self._plan = []
        self._runner = runner
        state = getattr(getattr(runner, "model", None),
                        "_poc_native_state", None)
        if scheduler_output is None:
            return
        # порядок как в _update_states резидуала: сначала уходы, потом приходы
        finished = getattr(scheduler_output, "finished_req_ids", ()) or ()
        for rid in list(self.reqs):
            if rid in finished:
                self._finalize(rid)
        # сироты: finished, доставленные на шаге без scheduled-токенов,
        # хуков не видят — реапим по runner.requests
        live = getattr(runner, "requests", None)
        if live is not None:
            for rid in list(self.reqs):
                if rid not in live:
                    self.reqs[rid].aborted = True
                    self._finalize(rid)
        self._register_new(scheduler_output)
        # преемпция: resume = откат num_computed — цепочка мертва
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        resumed = set(getattr(cached, "resumed_req_ids", ()) or ())
        for rid in list(self.reqs):
            if rid in resumed:
                self.reqs[rid].aborted = True
                self._finalize(rid)

        if not self.reqs:
            if self._mask_dirty and state is not None:
                state.mask.zero_()
                self._mask_dirty = False
            return
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
        order = sorted(((i, rid) for rid, i in idx_of.items()
                        if nsched.get(rid, 0) > 0))
        spans: Dict[str, tuple] = {}
        off = 0
        for _, rid in order:
            n = nsched[rid]
            spans[rid] = (off, n)
            off += n

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
                logger.error("engine-flow: %s foreign block_hash in step — "
                             "aborted", rid)
                req.aborted = True
                self._finalize(rid)
                continue
            if req.route_window != int(getattr(state, "route_window",
                                               req.route_window)):
                logger.error("engine-flow: %s route_window %s != engine %s — "
                             "aborted", rid, req.route_window,
                             getattr(state, "route_window", None))
                req.aborted = True
                self._finalize(rid)
                continue
            req.dev_init(dev)
            if req.route_seeds is None:
                req.route_seeds = [
                    _seed_from_string(route_base_seed(req.bh, req.nonce, li))
                    for li, _ in state._route_base]
            start, n = spans[rid]
            row_idx = idx_of[rid]
            computed = int(ib.num_computed_tokens_cpu[row_idx])
            if not req.prompt_done:
                npt = getattr(ib, "num_prompt_tokens", None)
                if npt is not None and int(npt[row_idx]) != req.seq_len:
                    logger.error("engine-flow: %s prompt len %s != seq_len %s"
                                 " — aborted", rid, int(npt[row_idx]),
                                 req.seq_len)
                    req.aborted = True
                    self._finalize(rid)
                    continue
                if req.prefill_embeds is None and computed > 0:
                    logger.error("engine-flow: %s prefix-cache hit at start — "
                                 "chain would be poisoned; aborted (run the "
                                 "server with --no-enable-prefix-caching)", rid)
                    req.aborted = True
                    self._finalize(rid)
                    continue
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
                    self._plan.append((rid, start + take - 1, 0))
            else:
                if computed < req.seq_len or n != 1:
                    # prompt_done, но префикс пересчитывается / не-декодный
                    # шаг: преемпция, которую resumed_req_ids не поймал
                    logger.error("engine-flow: %s recompute after preemption "
                                 "— aborted", rid)
                    req.aborted = True
                    self._finalize(rid)
                    continue
                step = req.step + 1
                if step > req.max_tokens:
                    continue
                emb = generate_decode_inputs_gpu(req.base_t, req.prev_k,
                                                 step, H, dev)
                sl = slice(start, start + 1)
                state.embeds[sl].copy_(emb.view(1, H).to(dt))
                state.mask[sl].fill_(True)
                self._route_span(state, req, sl, step=step)
                poc_rows.append(start)
                self._plan.append((rid, start, step))

        if poc_rows and bh is not None:
            self._set_reflections(state, bh)
            self._mask_dirty = True
        # прямые записи мимо кэш-ключей state — сбросить, чтобы выделенный
        # путь после нас не переиспользовал чужую раскладку
        state._rows_key = None
        state._route_key = None

    def _set_reflections(self, state, bh) -> None:
        # ВСЕГДА копируем (вектора кэшированы в _hh_cache, sha256 не
        # пересчитывается): выделенный путь между нашими шагами перезатирает
        # общие state.vectors, кэш-скип здесь дал бы wrong-k
        vs = state._hh_vectors(bh, None, state.vectors[0].dtype)
        for li in range(state.num_layers):
            state.vectors[li].copy_(vs[li].unsqueeze(0))

    def _route_span(self, state, req, sl, step: int) -> None:
        for (li, buf), seed in zip(state._route_base, req.route_seeds):
            buf[sl].fill_(seed)
        state.route_step[sl].fill_(int(step))

    # ------------------------------------------------------- post-forward --
    def post_forward(self, runner, scheduler_output, hidden_states) -> None:
        if not self._plan or not isinstance(hidden_states, torch.Tensor):
            return
        state = runner.model._poc_native_state
        dev = hidden_states.device
        if self._codebook is None:
            self._codebook = get_sphere_codebook().to(device=dev)
        finalized_all = False
        for rid, row, step in self._plan:
            req = self.reqs.get(rid)
            if req is None:
                continue
            h = hidden_states[row:row + 1].float()
            if step == 0:
                sph = random_pick_indices(req.bh, req.pk, [req.nonce],
                                          h.shape[1], SPHERE_DIM, dev)
            else:
                sph = random_pick_indices_gpu(req.base_t, req.prev_k, step,
                                              h.shape[1], SPHERE_DIM, dev)
            q = project_to_sphere(torch.gather(h, 1, sph))
            k, bad, margin = snap_with_margin(q, self._codebook)
            # цепочка целиком на устройстве — ни одного host-sync на шаге
            req.ks.append(k)
            req.margins.append(margin)
            req.prev_k = next_prev_k(k, None)
            req.step = step
            if step >= req.max_tokens:
                self._finalize(rid)
                finalized_all = not self.reqs
        if finalized_all and isinstance(
                getattr(runner.model, "_poc_native_state", None), object):
            state.mask.zero_()
            self._mask_dirty = False
        self._plan = []

    def _finalize(self, rid: str) -> None:
        req = self.reqs.pop(rid, None)
        if req is None:
            return
        ks = [int(t.item()) for t in req.ks]           # редкое событие —
        margins = [float(t.item()) for t in req.margins]  # sync допустим
        self.done.append({
            "req_id": rid,
            "nonce": req.nonce,
            "k_points_steps": ks,
            "margins_head": margins[:8],
            "n_steps": len(ks),
            "aborted": req.aborted or len(ks) < req.max_tokens + 1,
        })

    # ------------------------------------------------------------- drain --
    def collect(self, runner=None) -> dict:
        runner = runner or self._runner
        live = getattr(runner, "requests", None) if runner is not None else None
        if live is not None:
            for rid in list(self.reqs):
                if rid not in live:
                    self.reqs[rid].aborted = True
                    self._finalize(rid)
        with _LOCK:
            out = self.done
            self.done = []
        return {"artifacts": out, "in_flight": len(self.reqs)}


FLOW = EngineFlow()


def install(runner) -> bool:
    if os.environ.get("POC_ENGINE_FLOW", "0") != "1":
        return False
    pre = getattr(runner, "pre_forward_hooks", None)
    post = getattr(runner, "post_forward_hooks", None)
    if pre is None or post is None:
        raise RuntimeError(
            "engine-flow: residual lacks pre/post forward hook seams "
            "(need kaitakuai/vllm branch mixed/poc-as-request)")
    FLOW._runner = runner
    if FLOW.pre_forward not in pre:
        pre.append(FLOW.pre_forward)
    if FLOW.post_forward not in post:
        post.append(FLOW.post_forward)
    return True
