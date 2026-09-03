# SPDX-License-Identifier: Apache-2.0
"""Per-step PoC admission policy for the V1 scheduler.

All chat<->PoC mixing policy lives here so ``schedule()`` carries only a handful
of delegating calls. Construction is cheap and ``active`` is False whenever the
step holds no PoC request, in which case every method is a no-op and the
pure-chat path is untouched.

Policy (ported from the 0.20 in-tree branch, ``vllm/poc/mixed_decode.py``):
  * default is mixed batches (``policy.poc_mixed_batch``): a PoC prefill shares
    the step with decode; ``POC_MIXED_BATCH=0`` restores decode-only steps: chat
    and PoC share a forward only while BOTH are decoding, and either side's
    prefill runs isolated so the step lands on a captured cudagraph rung;
  * ``poc_share`` splits the step's token budget so PoC cannot starve chat;
  * ``poc_max_batch_size`` caps PoC rows per step;
  * a defer valve bounds consecutive chat-prefill defers so chat churn cannot
    starve a decoding PoC.
"""

from typing import TYPE_CHECKING

from gonka_poc.mixed.policy import (
    POC_DEFER_LIMIT,
    decode_only_mixing_gate,
    poc_alloc_footprint,
    poc_cfg,
    poc_share_budget,
    poc_mixed_batch,
    poc_step_num_tokens,
)
from gonka_poc.mixed.runtime import poc_kv_capacity, resolve_poc_max_batch_size
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _pool_diag(scheduler) -> str:
    """KV pool diagnostics at a stalled prefill (log only)."""
    try:
        km = scheduler.kv_cache_manager
        pool = km.block_pool
        free = pool.get_num_free_blocks()
        total = getattr(pool, "num_gpu_blocks", None)
        inflight = getattr(scheduler, "_inflight_prefills", ())
        reserved = scheduler._inflight_prefill_reserved_blocks() \
            if hasattr(scheduler, "_inflight_prefill_reserved_blocks") else -1
        need = -1
        wt = None
        for r in scheduler.waiting:
            if r.poc_params is not None:
                need = scheduler._request_remaining_blocks(r)
                wt = f"num_tokens={r.num_tokens} prompt={r.num_prompt_tokens}"
                break
        # KV groups: spec type and block size
        groups = []
        for g in getattr(km.kv_cache_config, "kv_cache_groups", ()):
            sp = g.kv_cache_spec
            groups.append(f"{type(sp).__name__}(bs={getattr(sp, 'block_size', '?')},"
                          f"layers={len(g.layer_names)},"
                          f"win={getattr(sp, 'sliding_window', getattr(sp, 'attention_chunk_size', None))})")
        # blocks held by the first few running PoC and chat rows
        held = []
        seen = {"poc": 0, "chat": 0}
        for r in scheduler.running:
            kind = "poc" if r.poc_params is not None else "chat"
            if seen[kind] >= 3:
                continue
            seen[kind] += 1
            n = 0
            for m in getattr(km.coordinator, "single_type_managers", ()):
                n += len(m.req_to_blocks.get(r.request_id, ()))
            held.append(f"{kind}:computed={r.num_computed_tokens},blocks={n}")
        return (f"free={free} total={total} inflight={len(inflight)} "
                f"reserved={reserved} need_first_waiting={need} [{wt}] "
                f"watermark={getattr(km, 'watermark_blocks', None)} "
                f"groups={' '.join(groups)} held={' '.join(held)}")
    except Exception as e:  # noqa: BLE001 — diagnostics must not break the step
        return f"diag failed: {e!r}"


def _install_alloc_diag(scheduler) -> None:
    """POC_DIAG=1: log allocate_slots refusals for PoC rows with pool state, at
    most once per second. Diagnostics only."""
    import os, time
    if os.environ.get("POC_DIAG", "") != "1":
        return
    km = scheduler.kv_cache_manager
    if getattr(km, "_poc_diag_wrapped", False):
        return
    orig = km.allocate_slots
    state = {"t": 0.0}

    def wrapped(request, num_new_tokens, *a, **kw):
        out = orig(request, num_new_tokens, *a, **kw)
        if out is None and getattr(request, "poc_params", None) is not None:
            now = time.monotonic()
            if now - state["t"] > 1.0:
                state["t"] = now
                logger.info("poc: alloc refused (new_tokens=%d computed=%d reserved=%s "
                            "running=%d waiting=%d; %s)", num_new_tokens,
                            request.num_computed_tokens, kw.get("reserved_blocks"),
                            len(scheduler.running), len(scheduler.waiting),
                            _pool_diag(scheduler))
        return out

    km.allocate_slots = wrapped
    km._poc_diag_wrapped = True


def _step_timer(scheduler, kind: str, prev_sched: int = -1, running: int = 0) -> None:
    """POC_DIAG=1: intervals between schedule() calls (= engine steps) per step
    kind; summary logged every 5 s. Diagnostics only."""
    import os, time
    if os.environ.get("POC_DIAG", "") != "1":
        return
    now = time.monotonic()
    st = getattr(scheduler, "_poc_step_stat", None)
    if st is None:
        st = scheduler._poc_step_stat = {"t": now, "t0": now, "d": {}}
        return
    d = st["d"].setdefault(kind, [])
    dt = (now - st["t"]) * 1000.0
    d.append(dt)
    st["t"] = now
    if 300.0 < dt < 5000.0:
        fin = getattr(scheduler, "finished_req_ids", None)
        logger.info("poc: slow step %.0f ms — prev sched=%d, running=%d, "
                    "waiting=%d, finished_in_step=%s",
                    dt, prev_sched, running, len(scheduler.waiting),
                    len(fin) if fin is not None else "?")
    comp = st.setdefault("comp", {})
    key = f"{kind}:sched={prev_sched}/run={running}"
    comp[key] = comp.get(key, 0) + 1
    if now - st["t0"] >= 5.0:
        parts = []
        for k, v in st["d"].items():
            v.sort()
            n = len(v)
            parts.append(f"{k}: n={n} mean={sum(v)/n:.1f} p50={v[n//2]:.1f} "
                         f"p90={v[int(n*.9)]:.1f} p99={v[int(n*.99)]:.1f} max={v[-1]:.1f}")
        comp = st.get("comp", {})
        top = sorted(comp.items(), key=lambda kv: -kv[1])[:6]
        logger.info("poc: steps(ms) %s || composition(prev sched/running: n) %s",
                    " | ".join(parts), " ".join(f"{k}:{v}" for k, v in top))
        st["t0"] = now
        st["d"] = {}
        st["comp"] = {}


def _kv_headroom_allow(scheduler):
    """How many PoC prefills fit into this step while leaving headroom in the
    pool: one block per running row (growth this step) plus a POC_KV_HEADROOM
    fraction (default 1%) of the pool. None: vLLM internals unavailable, gate
    off. Checking the first row alone is not enough: a burst of up to
    MNBT/seq_len prefills in one step overruns any headroom."""
    import os
    try:
        km = scheduler.kv_cache_manager
        free = km.block_pool.get_num_free_blocks()
        total = km.block_pool.num_gpu_blocks
        need = 0
        for r in scheduler.waiting:
            if r.poc_params is not None:
                need = scheduler._request_remaining_blocks(r)
                break
        if need <= 0:
            return None
        frac = float(os.environ.get("POC_KV_HEADROOM", "0.01") or 0.0)
        reserve = len(scheduler.running) + int(total * frac)
        return max(0, (free - reserve) // need)
    except Exception:  # noqa: BLE001 — gate helper, must not break the step
        return None


def _prefill_per_step() -> int:
    import os
    try:
        return int(os.environ.get("POC_PREFILL_PER_STEP", "0") or 0)
    except ValueError:
        return 0


class PoCAdmission:
    """Decides which PoC/chat requests may enter the current forward."""

    __slots__ = ("active", "_defer_chat", "_defer_poc", "_max_batch",
                 "_token_budget", "_scheduled", "_tokens", "_poc_prefill",
                 "_scheduler", "_prefill_landing", "_any_scheduled", "_stalled",
                 "_new_prefills", "_prefill_allow", "_step_budget", "_all_tokens")

    # Steps to keep the step decode-only after a stalled prefill before retrying
    # prefill (unless a row finishes earlier).
    STALL_RETRY_STEPS = 16

    def __init__(self, scheduler, token_budget: int) -> None:
        # skipped_waiting: rows deferred by skip() last step; without them the
        # scan misses the pending prefill and drops the gates.
        queues = (scheduler.running, scheduler.waiting,
                  getattr(scheduler, "skipped_waiting", None) or ())
        # ONE pass for all four flags. As four separate any() calls, the ones whose
        # answer is "no" walk the whole of running+waiting every step — and on a
        # PoC-only node both chat questions are exactly that, so the queues were
        # scanned twice per step for nothing.
        poc_present = chat_present = False
        poc_will_prefill = chat_will_prefill = False
        chat_running = 0
        for r in scheduler.running:
            if r.poc_params is None:
                chat_running += 1
        for q in queues:
            for r in q:
                if r.poc_params is not None:
                    poc_present = True
                    if r.num_computed_tokens == 0:
                        poc_will_prefill = True
                else:
                    chat_present = True
                    if r.num_computed_tokens < r.num_prompt_tokens:
                        chat_will_prefill = True
                if (poc_present and chat_present
                        and poc_will_prefill and chat_will_prefill):
                    break
            else:
                continue
            break
        self.active = poc_present
        _p0 = getattr(scheduler, "_poc_admission", None)
        _step_timer(scheduler, "poc" if poc_present else "chat",
                    getattr(_p0, "_scheduled", -1) if (_p0 is not None and _p0.active) else -1,
                    len(scheduler.running))
        if not self.active:
            return

        _install_alloc_diag(scheduler)
        _prev = getattr(scheduler, "_poc_admission", None)
        if (_prev is not None and _prev.active and poc_will_prefill
                and _prev._new_prefills == 0):
            import os, time
            if os.environ.get("POC_DIAG", "") == "1":
                _t = getattr(scheduler, "_poc_diag_t", 0.0)
                if time.monotonic() - _t > 1.0:
                    scheduler._poc_diag_t = time.monotonic()
                    logger.info(
                        "poc: diag — waiting rows, no prefill taken: running=%d "
                        "waiting=%d prev(defer_poc=%s defer_chat=%s scheduled=%d "
                        "max_batch=%d tokens=%d budget=%d stalled=%s landing=%d "
                        "poc_prefill=%s any=%d) free=%d",
                        len(scheduler.running), len(scheduler.waiting),
                        _prev._defer_poc, _prev._defer_chat, _prev._scheduled,
                        _prev._max_batch, _prev._tokens, _prev._token_budget,
                        _prev._stalled, len(_prev._prefill_landing),
                        _prev._poc_prefill, _prev._any_scheduled,
                        scheduler.kv_cache_manager.block_pool.get_num_free_blocks())
        cache_config = scheduler.cache_config
        # Resolve here, not at config init: num_gpu_blocks is only known once
        # the engine has profiled free memory and built the KV pool.
        # num_gpu_blocks*block_size only measures the pool for a homogeneous KV.
        # Under hybrid KV (DeepSeek V4: full attention + 128/8 windows with
        # 256/64/8/4 blocks) vLLM puts the smallest group's block size into
        # cache_config.block_size, so the formula undercounts the pool by an
        # order of magnitude and caps PoC well below what actually runs; the
        # excess rows starve until the admitted batch ends. Memory admission is already
        # covered by vLLM's full_sequence_must_fit plus the livelock guard below,
        # so the KV clamp is skipped under hybrid KV.
        kv_groups = getattr(getattr(getattr(scheduler, "kv_cache_manager", None),
                                    "kv_cache_config", None), "kv_cache_groups", None)
        hybrid_kv = kv_groups is not None and len(kv_groups) > 1
        kv_capacity = 0 if hybrid_kv else poc_kv_capacity(
            getattr(cache_config, "num_gpu_blocks", 0),
            getattr(cache_config, "block_size", 0),
            poc_cfg(cache_config, "poc_seq_len"),
            poc_cfg(cache_config, "poc_max_tokens"),
        )
        self._max_batch = resolve_poc_max_batch_size(
            poc_cfg(cache_config, "poc_max_batch_size"),
            scheduler.scheduler_config.max_num_seqs,
            kv_capacity,
        )
        # A PoC decode step must land on a captured CUDA graph: a batch above
        # max_cudagraph_capture_size runs eager, roughly 2x slower per step.
        cg = getattr(getattr(getattr(scheduler, "vllm_config", None),
                             "compilation_config", None),
                     "max_cudagraph_capture_size", None)
        if cg and not poc_cfg(cache_config, "poc_max_batch_size"):
            # The graph captures the WHOLE step, chat rows included: chat plus
            # PoC above the capture size pushes the step to eager and both lose
            # most of their speed. Cap PoC at what is left of the graph after
            # the running chat rows.
            self._max_batch = max(1, min(self._max_batch, int(cg) - chat_running))
        if not getattr(scheduler, "_poc_max_batch_logged", False):
            scheduler._poc_max_batch_logged = True
            logger.info("poc: PoC rows per step capped at %d (hybrid_kv=%s, "
                        "kv_capacity=%d, cudagraph_max=%s, configured=%d, "
                        "max_num_seqs=%d)",
                        self._max_batch, hybrid_kv, kv_capacity, cg,
                        poc_cfg(cache_config, "poc_max_batch_size"),
                        scheduler.scheduler_config.max_num_seqs)
        self._token_budget = poc_share_budget(
            poc_cfg(cache_config, "poc_share"), token_budget, chat_present)
        self._scheduled = 0
        self._tokens = 0
        self._any_scheduled = 0
        self._new_prefills = 0
        # Whole-step budget and what is already taken in it (chat and PoC): a
        # PoC prefill is all-or-nothing (seq_len) and the scheduler has already
        # clamped num_new_tokens to the remaining budget by then; num_tokens()
        # returns seq_len over that clamp, so without this check the step
        # overflowed (assert token_budget >= 0 in schedule(), EngineCore death).
        self._step_budget = int(token_budget)
        self._all_tokens = 0

        # Liveness. If the previous step announced a PoC prefill but scheduled
        # NO rows at all — waiting rows lacked KV and decode rows were held for
        # decode-only steps — the engine spins forever: only decode frees memory and
        # decode waits for the prefill. So the step goes to decode (waiting rows
        # held, step stays uniform). Prefill is retried once any row finishes
        # (running shrank) or every STALL_RETRY_STEPS.
        # KV headroom: do not admit a new PoC prefill if the pool would then
        # lack room for running rows to grow (one block each) plus the reserve
        # fraction. Otherwise the pool hits 100% and vLLM preempts rows. Waiting
        # rows just wait and the step goes to decode; prefill resumes as rows
        # finish.
        self._prefill_allow = None
        if poc_will_prefill:
            self._prefill_allow = _kv_headroom_allow(scheduler)
            # POC_PREFILL_PER_STEP=k: at most k new PoC prefills per step. Useful
            # with mixed batches: a small prefill slice inside a decode step keeps
            # the step within the graph size instead of a 16k-token eager step.
            k = _prefill_per_step()
            if k > 0:
                self._prefill_allow = k if self._prefill_allow is None \
                    else min(self._prefill_allow, k)
            if self._prefill_allow == 0:
                poc_will_prefill = False
        prev = getattr(scheduler, "_poc_admission", None)
        stalled = False
        if poc_will_prefill:
            if (prev is not None and prev.active and prev._poc_prefill
                    and prev._any_scheduled == 0):
                if getattr(scheduler, "_poc_stall_running", None) is None:
                    logger.info(
                        "poc: stall — PoC prefill did not fit in KV, step given "
                        "to decode (running=%d, waiting=%d; %s)",
                        len(scheduler.running), len(scheduler.waiting),
                        _pool_diag(scheduler))
                scheduler._poc_stall_running = len(scheduler.running)
                scheduler._poc_stall_steps = 0
                stalled = True
            elif getattr(scheduler, "_poc_stall_running", None) is not None:
                steps = getattr(scheduler, "_poc_stall_steps", 0) + 1
                if (len(scheduler.running) < scheduler._poc_stall_running
                        or steps >= self.STALL_RETRY_STEPS):
                    scheduler._poc_stall_running = None  # retry prefill
                else:
                    scheduler._poc_stall_steps = steps
                    stalled = True
        self._stalled = stalled
        scheduler._poc_admission = self
        if stalled:
            poc_will_prefill = False
        # Deferred first decode step. Prefill publishes the nonce's prev_k, but
        # num_computed_tokens advances at SCHEDULING time, so a row reaches
        # decode before its prefill output has landed. The async scheduler keeps
        # a queue of depth 1: step N output is processed by step N+2. A delay of
        # exactly one step closes the race. With rolling admission the
        # prefill->decode seam happens on every refill, so this is mandatory.
        self._scheduler = scheduler
        self._prefill_landing = getattr(scheduler, "_poc_prefill_landing", None) or set()
        scheduler._poc_prefill_landing = set()

        # Vestigial in the 0.20 branch (hardcoded False): PoC never demands an
        # exclusive pure-decode step, so chat+PoC decode freely share a forward.
        poc_decode_pending = False
        self._poc_prefill = poc_will_prefill
        self._defer_chat, self._defer_poc, scheduler._poc_defers = (
            decode_only_mixing_gate(
                # Mixed batches: no prefill isolation (the gate's original behaviour).
                mixed_cudagraph=not poc_mixed_batch(),
                poc_decode_pending=poc_decode_pending,
                poc_will_prefill=poc_will_prefill,
                chat_will_prefill=chat_will_prefill,
                consecutive_defers=getattr(scheduler, "_poc_defers", 0),
                defer_limit=POC_DEFER_LIMIT,
            )
        )

    def skip(self, request: "Request") -> bool:
        """True if this request must not be scheduled into this step."""
        if not self.active:
            return False
        if request.poc_params is None:
            return self._defer_chat
        if self._defer_poc or self._scheduled >= self._max_batch:
            return True
        # Step given to decode after a stalled prefill: waiting rows wait until
        # decode frees KV (step stays decode-only).
        if request.num_computed_tokens == 0 and (
                self._stalled or (self._prefill_allow is not None
                                  and self._new_prefills >= self._prefill_allow)):
            return True
        # This row's prefill landed last step; its output is still in flight.
        if getattr(request, "request_id", None) in self._prefill_landing:
            return True
        # Mixed batches: PoC prefill and decode rows share the step. Otherwise
        # decode-only steps (a pure decode step lands on a captured CUDA graph).
        if poc_mixed_batch():
            return False
        # Keep a step uniform: never mix a PoC prefill with PoC decode rows.
        return self._poc_prefill and request.num_computed_tokens > 0

    def num_tokens(self, request: "Request", num_new_tokens: int) -> int:
        """Token count for a PoC row; chat rows pass through unchanged."""
        if not self.active or request.poc_params is None:
            return num_new_tokens
        return poc_step_num_tokens(request.poc_params, request.num_computed_tokens)

    def over_budget(self, request: "Request", num_new_tokens: int) -> bool:
        """True once PoC has consumed its slice of this step's token budget, or
        when the whole step cannot take these tokens on top of what chat and
        PoC already scheduled (a PoC prefill is all-or-nothing)."""
        if not self.active or request.poc_params is None:
            return False
        if self._tokens + num_new_tokens > self._token_budget:
            return True
        return self._all_tokens + num_new_tokens > self._step_budget

    def alloc_tokens(self, request: "Request", num_new_tokens: int) -> int:
        """KV footprint to reserve via the shared KVCacheManager."""
        if not self.active or request.poc_params is None:
            return num_new_tokens
        return poc_alloc_footprint(request.poc_params, num_new_tokens)

    def note_scheduled(self, request: "Request", num_new_tokens: int) -> None:
        if not self.active:
            return
        self._any_scheduled += 1
        self._all_tokens += int(num_new_tokens)
        if request.poc_params is None:
            return
        self._scheduled += 1
        if request.num_computed_tokens == 0:
            self._new_prefills += 1
        self._tokens += num_new_tokens
        # Rows whose prefill COMPLETES in this step: their output lands only
        # after the next step is scheduled, so skip() holds their decode one step.
        seq_len = getattr(request.poc_params, "seq_len", None)
        rid = getattr(request, "request_id", None)
        done = getattr(request, "num_computed_tokens", None)
        if seq_len and rid is not None and done is not None \
                and done < seq_len <= done + num_new_tokens:
            self._scheduler._poc_prefill_landing.add(rid)
