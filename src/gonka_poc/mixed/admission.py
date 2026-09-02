# SPDX-License-Identifier: Apache-2.0
"""Per-step PoC admission policy for the V1 scheduler.

All chat<->PoC mixing policy lives here so ``schedule()`` carries only a handful
of delegating calls. Construction is cheap and ``active`` is False whenever the
step holds no PoC request, in which case every method is a no-op and the
pure-chat path is untouched.

Policy (ported from the 0.20 in-tree branch, ``vllm/poc/mixed_decode.py``):
  * chat and PoC share a forward only while BOTH are decoding; either side's
    prefill runs isolated, so a mixed step stays uniform-decode shaped and lands
    on a captured cudagraph rung;
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
    poc_chat_like,
    poc_step_num_tokens,
)
from gonka_poc.mixed.runtime import poc_kv_capacity, resolve_poc_max_batch_size
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


def _pool_diag(scheduler) -> str:
    """Диагностика пула KV в момент застрявшего префилла (только для лога)."""
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
        # группы KV: тип спецификации и размер блока
        groups = []
        for g in getattr(km.kv_cache_config, "kv_cache_groups", ()):
            sp = g.kv_cache_spec
            groups.append(f"{type(sp).__name__}(bs={getattr(sp, 'block_size', '?')},"
                          f"layers={len(g.layer_names)},"
                          f"win={getattr(sp, 'sliding_window', getattr(sp, 'attention_chunk_size', None))})")
        # сколько блоков держат первые бегущие строки PoC и чата
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
    except Exception as e:  # noqa: BLE001 — диагностика не должна ронять шаг
        return f"diag failed: {e!r}"


def _install_alloc_diag(scheduler) -> None:
    """POC_DIAG=1: логировать (не чаще раза в секунду) отказ allocate_slots
    для PoC-строки с состоянием пула. Только диагностика."""
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
                logger.info("poc: alloc отказ (new_tokens=%d computed=%d reserved=%s "
                            "running=%d waiting=%d; %s)", num_new_tokens,
                            request.num_computed_tokens, kw.get("reserved_blocks"),
                            len(scheduler.running), len(scheduler.waiting),
                            _pool_diag(scheduler))
        return out

    km.allocate_slots = wrapped
    km._poc_diag_wrapped = True


def _step_timer(scheduler, kind: str, prev_sched: int = -1, running: int = 0) -> None:
    """POC_DIAG=1: интервалы между вызовами schedule() (= шаги движка) по видам
    шага; сводка в лог каждые 5 с. Только диагностика."""
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
        logger.info("poc: долгий шаг %.0f мс — prev sched=%d, running=%d, "
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
        logger.info("poc: шаги(мс) %s || состав(prev sched/running: n) %s",
                    " | ".join(parts), " ".join(f"{k}:{v}" for k, v in top))
        st["t0"] = now
        st["d"] = {}
        st["comp"] = {}


def _kv_headroom_allow(scheduler):
    """Сколько префиллов PoC можно посадить в этот шаг, чтобы после них в пуле
    остался запас: по блоку на каждую бегущую строку (рост на шаге) плюс доля
    POC_KV_HEADROOM (по умолчанию 1%) от пула. None — внутренности vLLM
    недоступны, гейт выключен. Проверка на первой строке недостаточна: волна до
    MNBT/seq_len префиллов в одном шаге (63 на H100) перекрывает любой запас."""
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
    except Exception:  # noqa: BLE001 — гейт-помощник, не должен ронять шаг
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
                 "_new_prefills", "_prefill_allow")

    # Сколько шагов держать шаг за декодом после застрявшего префилла, прежде
    # чем снова попробовать префилл (если ни одна строка не завершилась раньше).
    STALL_RETRY_STEPS = 16

    def __init__(self, scheduler, token_budget: int) -> None:
        queues = (scheduler.running, scheduler.waiting)
        # ONE pass for all four flags. As four separate any() calls, the ones whose
        # answer is "no" walk the whole of running+waiting every step — and on a
        # PoC-only node both chat questions are exactly that, so the queues were
        # scanned twice per step for nothing.
        poc_present = chat_present = False
        poc_will_prefill = chat_will_prefill = False
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
                        "poc: диаг — ожидающие есть, префилл не взят: running=%d "
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
        # Ёмкость пула по формуле num_gpu_blocks*block_size имеет смысл только
        # для однородного KV. Под гибридным KV (DeepSeek V4: полное внимание +
        # окна 128/8 с блоками 256/64/8/4) vLLM кладёт в cache_config.block_size
        # блок самой мелкой группы, и формула занижает пул в десятки раз:
        # на 4×H100 кап выходил 134 строки при бегущих 197, лишние строки
        # голодали до конца кохорты (02.09.2026). Допуск по памяти и так
        # держит full_sequence_must_fit у vLLM плюс защита от лайвлока ниже,
        # поэтому под гибридом клэмп по KV не применяем.
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
        # Декодный шаг PoC должен ложиться на захваченный CUDA-граф: батч больше
        # max_cudagraph_capture_size исполняется eager (на 4×H100: 600 строк —
        # 133 мс/шаг против 71 мс у чата на 512 строках, 02.09.2026).
        cg = getattr(getattr(getattr(scheduler, "vllm_config", None),
                             "compilation_config", None),
                     "max_cudagraph_capture_size", None)
        if cg and not poc_cfg(cache_config, "poc_max_batch_size"):
            self._max_batch = min(self._max_batch, int(cg))
        if not getattr(scheduler, "_poc_max_batch_logged", False):
            scheduler._poc_max_batch_logged = True
            logger.info("poc: строк PoC на шаг не более %d (hybrid_kv=%s, "
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

        # Живость (лайвлок на Hopper, 01.09.2026). Если прошлый шаг объявил
        # префилл PoC, но не запланировал НИ ОДНОЙ строки — ожидающим не хватило
        # KV, а декодные были удержаны ради uniform-step, — движок крутится
        # вхолостую навсегда: память освобождает только декод, а декод ждёт
        # префилла. Тогда шаг отдаётся декоду (ожидающие строки удерживаются,
        # шаг остаётся однородным). Префилл пробуем снова, когда хоть одна
        # строка завершилась (running уменьшился) или каждые STALL_RETRY_STEPS.
        # Запас KV: не сажать новый префилл PoC, если после него пулу не хватит
        # на рост бегущих строк (по блоку на строку) и на долю запаса.
        # Иначе банкет доводит пул до 100% и vLLM вытесняет строки (2 вытеснения
        # за прогон 600 при капе 512, 02.09.2026). Ожидающие просто ждут, шаг
        # отдаётся декоду — префилл вернётся, когда строки завершатся.
        self._prefill_allow = None
        if poc_will_prefill:
            self._prefill_allow = _kv_headroom_allow(scheduler)
            # POC_PREFILL_PER_STEP=k: не больше k новых префиллов PoC за шаг.
            # Смысл в режиме как-чат: маленькая порция префилла в декодном шаге
            # держит шаг в размере графа вместо eager-волны на 16k токенов.
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
                        "poc: stall — префилл PoC не влез в KV, шаг отдан "
                        "декоду (running=%d, waiting=%d; %s)",
                        len(scheduler.running), len(scheduler.waiting),
                        _pool_diag(scheduler))
                scheduler._poc_stall_running = len(scheduler.running)
                scheduler._poc_stall_steps = 0
                stalled = True
            elif getattr(scheduler, "_poc_stall_running", None) is not None:
                steps = getattr(scheduler, "_poc_stall_steps", 0) + 1
                if (len(scheduler.running) < scheduler._poc_stall_running
                        or steps >= self.STALL_RETRY_STEPS):
                    scheduler._poc_stall_running = None  # снова пробуем префилл
                else:
                    scheduler._poc_stall_steps = steps
                    stalled = True
        self._stalled = stalled
        scheduler._poc_admission = self
        if stalled:
            poc_will_prefill = False
        # Отложенный первый шаг декода (возврат fafabbd). Префилл публикует
        # prev_k нонса, а num_computed_tokens продвигается в момент ПЛАНИРОВАНИЯ,
        # так что строка доходит до декода раньше, чем сел вывод префилла.
        # Асинхронный планировщик держит очередь глубины 1: вывод шага N
        # обработан к шагу N+2. Задержка ровно на один шаг закрывает гонку.
        # При скользящей подаче стык префилл->декод случается на каждой волне,
        # поэтому страховка обязательна, а не «ради редкого случая».
        self._scheduler = scheduler
        self._prefill_landing = getattr(scheduler, "_poc_prefill_landing", None) or set()
        scheduler._poc_prefill_landing = set()

        # Vestigial in the 0.20 branch (hardcoded False): PoC never demands an
        # exclusive pure-decode step, so chat+PoC decode freely share a forward.
        poc_decode_pending = False
        self._poc_prefill = poc_will_prefill
        self._defer_chat, self._defer_poc, scheduler._poc_defers = (
            decode_only_mixing_gate(
                # Как чат: без изоляции префиллов (исходное поведение гейта).
                mixed_cudagraph=not poc_chat_like(),
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
        # Шаг отдан декоду после застрявшего префилла: ожидающие строки ждут,
        # пока декод освободит KV (шаг остаётся однородным).
        if request.num_computed_tokens == 0 and (
                self._stalled or (self._prefill_allow is not None
                                  and self._new_prefills >= self._prefill_allow)):
            return True
        # Префилл этой строки сел в прошлом шаге — её вывод ещё в полёте.
        if getattr(request, "request_id", None) in self._prefill_landing:
            return True
        # Как чат: префилл и декод PoC-строк делят шаг. Иначе — uniform-step
        # (чисто декодный шаг ложится на захваченный CUDA-граф).
        if poc_chat_like():
            return False
        # Keep a step uniform: never mix a PoC prefill with PoC decode rows.
        return self._poc_prefill and request.num_computed_tokens > 0

    def num_tokens(self, request: "Request", num_new_tokens: int) -> int:
        """Token count for a PoC row; chat rows pass through unchanged."""
        if not self.active or request.poc_params is None:
            return num_new_tokens
        return poc_step_num_tokens(request.poc_params, request.num_computed_tokens)

    def over_budget(self, request: "Request", num_new_tokens: int) -> bool:
        """True once PoC has consumed its slice of this step's token budget."""
        if not self.active or request.poc_params is None:
            return False
        return self._tokens + num_new_tokens > self._token_budget

    def alloc_tokens(self, request: "Request", num_new_tokens: int) -> int:
        """KV footprint to reserve via the shared KVCacheManager."""
        if not self.active or request.poc_params is None:
            return num_new_tokens
        return poc_alloc_footprint(request.poc_params, num_new_tokens)

    def note_scheduled(self, request: "Request", num_new_tokens: int) -> None:
        if not self.active:
            return
        self._any_scheduled += 1
        if request.poc_params is None:
            return
        self._scheduled += 1
        if request.num_computed_tokens == 0:
            self._new_prefills += 1
        self._tokens += num_new_tokens
        # Строки, чей префилл ЗАВЕРШАЕТСЯ в этом шаге: их вывод сядет только
        # после планирования следующего, skip() удержит их декод на один шаг.
        seq_len = getattr(request.poc_params, "seq_len", None)
        rid = getattr(request, "request_id", None)
        done = getattr(request, "num_computed_tokens", None)
        if seq_len and rid is not None and done is not None \
                and done < seq_len <= done + num_new_tokens:
            self._scheduler._poc_prefill_landing.add(rid)
