"""Чистая логика допуска PoC на фальшивом планировщике (без GPU и без vLLM-движка)."""
import types

from gonka_poc.mixed.admission import PoCAdmission


class Req:
    def __init__(self, rid, poc, computed=0, prompt=256):
        self.request_id = rid
        self.poc_params = types.SimpleNamespace(seq_len=256, max_tokens=256) if poc else None
        self.num_computed_tokens = computed
        self.num_prompt_tokens = prompt


def make_sched(running=(), waiting=(), skipped=(), free=10000, total=17278, cg=512, groups=2, need=105):
    s = types.SimpleNamespace()
    s.running, s.waiting, s.skipped_waiting = list(running), list(waiting), list(skipped)
    s.cache_config = types.SimpleNamespace(num_gpu_blocks=total, block_size=4, poc_max_batch_size=0,
                                           poc_seq_len=256, poc_max_tokens=256, poc_share=0.5)
    s.scheduler_config = types.SimpleNamespace(max_num_seqs=1024)
    s.kv_cache_manager = types.SimpleNamespace(
        block_pool=types.SimpleNamespace(get_num_free_blocks=lambda: free, num_gpu_blocks=total),
        kv_cache_config=types.SimpleNamespace(kv_cache_groups=[object()] * groups),
        watermark_blocks=0)
    s.vllm_config = types.SimpleNamespace(compilation_config=types.SimpleNamespace(max_cudagraph_capture_size=cg))
    s._request_remaining_blocks = lambda r: need
    s._inflight_prefills = set()
    s._inflight_prefill_reserved_blocks = lambda: 0
    s.finished_req_ids = set()
    return s


def test_cap_is_cudagraph_minus_chat_rows():
    chat = [Req(f"c{i}", False, computed=300) for i in range(256)]
    poc = [Req(f"p{i}", True, computed=300) for i in range(10)]
    a = PoCAdmission(make_sched(running=chat + poc), token_budget=16384)
    assert a._max_batch == 512 - 256
    a2 = PoCAdmission(make_sched(running=poc, cg=512), token_budget=16384)
    assert a2._max_batch == 512


def test_hybrid_kv_disables_formula_clamp():
    poc = [Req("p1", True, computed=300)]
    hybrid = PoCAdmission(make_sched(running=poc, groups=5, cg=None), token_budget=16384)
    assert hybrid._max_batch == 1024
    single = PoCAdmission(make_sched(running=poc, groups=1, cg=None, total=1000), token_budget=16384)
    assert single._max_batch == 1000 * 4 // 512


def test_headroom_limits_prefills_per_step(monkeypatch):
    monkeypatch.setenv("POC_KV_HEADROOM", "0.0")
    running = [Req(f"p{i}", True, computed=300) for i in range(20)]
    waiting = [Req(f"w{i}", True, computed=0) for i in range(5)]
    # free 340 − reserve 20 = 320 → 3 префилла по 105
    a = PoCAdmission(make_sched(running=running, waiting=waiting, free=340), token_budget=16384)
    assert a._prefill_allow == 3
    admitted = 0
    for w in waiting:
        if not a.skip(w):
            a.note_scheduled(w, 256); admitted += 1
    assert admitted == 3


def test_stall_hands_next_step_to_decode(monkeypatch):
    monkeypatch.setenv("POC_CHAT_LIKE", "0")
    running = [Req(f"p{i}", True, computed=300) for i in range(3)]
    waiting = [Req("w0", True, computed=0)]
    s = make_sched(running=running, waiting=waiting)
    a = PoCAdmission(s, token_budget=16384)
    assert a._poc_prefill and all(a.skip(r) for r in running)   # uniform-step держит декод
    # шаг ничего не запланировал → следующий шаг отдаётся декоду
    b = PoCAdmission(s, token_budget=16384)
    assert b._stalled and not b._poc_prefill
    assert not any(b.skip(r) for r in running) and b.skip(waiting[0])


def test_prefill_per_step_knob(monkeypatch):
    monkeypatch.setenv("POC_PREFILL_PER_STEP", "2")
    waiting = [Req(f"w{i}", True, computed=0) for i in range(6)]
    a = PoCAdmission(make_sched(waiting=waiting, free=100000), token_budget=16384)
    n = 0
    for w in waiting:
        if not a.skip(w):
            a.note_scheduled(w, 256); n += 1
    assert n == 2


def test_step_budget_guard_refuses_partial_prefill():
    chat = [Req("c0", False, computed=0, prompt=16300)]
    w = Req("w0", True, computed=0)
    a = PoCAdmission(make_sched(running=chat, waiting=[w], free=100000), token_budget=16384)
    a.note_scheduled(chat[0], 16300)                 # чат съел почти весь шаг
    assert a.num_tokens(w, 84) == 256                # планировщик сжал до остатка, PoC хочет целиком
    assert a.over_budget(w, 256)                     # ... и должен быть отвергнут
    b = PoCAdmission(make_sched(waiting=[w], free=100000), token_budget=16384)
    assert not b.over_budget(w, 256)


def test_skipped_waiting_is_scanned():
    running = [Req("p0", True, computed=300)]
    skipped = [Req("w0", True, computed=0)]
    a = PoCAdmission(make_sched(running=running, skipped=skipped, free=100000), token_budget=16384)
    assert a._poc_prefill                            # префилл из skipped_waiting виден
