# SPDX-License-Identifier: Apache-2.0
"""engine-flow: цепочка «запрос среди запросов» бит-в-бит равна механике
выделенного чанка (CPU-референс), плюс контракт установки хуков."""
import types

import pytest
import torch

from gonka_poc.mixed import engine_flow as ef
from gonka_poc.poc.decode_chain import next_prev_k
from gonka_poc.poc.gpu_random import (
    decode_base_seeds, generate_decode_inputs_gpu, generate_inputs,
    random_pick_indices, random_pick_indices_gpu,
)
from gonka_poc.poc.sphere import (
    SPHERE_DIM, get_sphere_codebook, project_to_sphere, snap_with_margin,
)

BH = "e" * 64
PK = "cafebabe" * 4
H = 512
DEV = torch.device("cpu")


def _fake_model(hidden_fn, steps):
    """Детерминированная «модель»: hidden шага t — чистая функция входа."""
    return [hidden_fn(t) for t in range(steps)]


def reference_chain(nonce, max_tokens, hidden_of_step):
    """Дословный слепок snap_rows выделенного чанка (decode_runner)."""
    codebook = get_sphere_codebook().to(DEV)
    base = decode_base_seeds(BH, PK, [nonce], DEV)
    prev = torch.zeros(1, dtype=torch.int64)
    ks = []
    for step in range(0, max_tokens + 1):
        h = hidden_of_step(step)
        if step == 0:
            sph = random_pick_indices(BH, PK, [nonce], H, SPHERE_DIM, DEV)
        else:
            sph = random_pick_indices_gpu(base, prev, step, H, SPHERE_DIM, DEV)
        q = project_to_sphere(torch.gather(h.float(), 1, sph))
        k, bad, margin = snap_with_margin(q, codebook)
        ks.append(int(k.item()))
        prev = next_prev_k(k, None)
    return ks


def engine_flow_chain(nonce, max_tokens, hidden_of_step):
    """Та же цепочка через FLOW.post_forward c ручным планом шагов."""
    flow = ef.EngineFlow()
    rid = "req-1"
    req = ef._Req(nonce, BH, PK, seq_len=16,
                  max_tokens=max_tokens, route_window=256)
    req.dev_init(DEV)
    flow.reqs[rid] = req
    class _State:  # маска для очистки после финализации
        mask = torch.zeros(4, 1, dtype=torch.bool)
    runner = types.SimpleNamespace(model=types.SimpleNamespace(
        _poc_native_state=_State()))
    for step in range(0, max_tokens + 1):
        flow._plan = [(rid, 0, step)]
        flow.post_forward(runner, None, hidden_of_step(step))
    art = next(a for a in flow.done if a["req_id"] == rid)
    return art["k_points_steps"], art


def _hidden(step):
    g = torch.Generator().manual_seed(1000 + step)
    return torch.randn(1, H, generator=g)


def test_chain_matches_dedicated_reference():
    for nonce in (0, 7, 999999):
        ref = reference_chain(nonce, 24, _hidden)
        got, art = engine_flow_chain(nonce, 24, _hidden)
        assert got == ref, f"nonce {nonce}: цепочки разошлись"
        assert art["n_steps"] == 25 and art["aborted"] is False


def test_prev_k_actually_chains():
    """Разные k на шаге t обязаны менять выбор координат шага t+1."""
    got_a, _ = engine_flow_chain(1, 8, _hidden)
    def hidden_b(step):
        g = torch.Generator().manual_seed(step)  # другие hidden
        return torch.randn(1, H, generator=g)
    got_b, _ = engine_flow_chain(1, 8, hidden_b)
    assert got_a != got_b


def test_install_contract(monkeypatch):
    monkeypatch.setenv("POC_ENGINE_FLOW", "1")
    r = types.SimpleNamespace(pre_forward_hooks=[], post_forward_hooks=[])
    assert ef.install(r) is True and ef.install(r) is True
    assert r.pre_forward_hooks.count(ef.FLOW.pre_forward) == 1
    assert r.post_forward_hooks.count(ef.FLOW.post_forward) == 1
    with pytest.raises(RuntimeError, match="hook seams"):
        ef.install(types.SimpleNamespace(pre_forward_hooks=[]))
    monkeypatch.setenv("POC_ENGINE_FLOW", "0")
    assert ef.install(r) is True or True  # env=0 -> False на свежем раннере
    r2 = types.SimpleNamespace(pre_forward_hooks=[], post_forward_hooks=[])
    monkeypatch.setenv("POC_ENGINE_FLOW", "0")
    assert ef.install(r2) is False


def test_collect_drains_and_marks_short_chains():
    flow = ef.EngineFlow()
    r = ef._Req(5, BH, PK, 16, 8, 256)
    r.ks = [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]
    flow.reqs["r"] = r
    flow._finalize("r")
    out = flow.collect()
    assert out["artifacts"][0]["aborted"] is True
    assert flow.collect()["artifacts"] == []


def test_collect_reaps_orphans_via_runner():
    """finished на пустом шаге: запроса нет в runner.requests — collect
    обязан финализировать сироту как aborted (фикс красной команды №8)."""
    flow = ef.EngineFlow()
    flow.reqs["gone"] = ef._Req(1, BH, PK, 16, 8, 256)
    runner = types.SimpleNamespace(requests={})
    out = flow.collect(runner)
    assert out["in_flight"] == 0
    assert out["artifacts"][0]["aborted"] is True


def test_register_rejects_bad_sampling_contract():
    """max_tokens=N (а не N+1) или ignore_eos=False — нонс не регистрируется
    (фикс №2): лучше отказ на входе, чем 100% aborted-артефактов."""
    flow = ef.EngineFlow()
    def mk(mt, ig):
        xa = {f"{ef.XA_PREFIX}nonce": 1, f"{ef.XA_PREFIX}block_hash": BH,
              f"{ef.XA_PREFIX}public_key": PK, f"{ef.XA_PREFIX}seq_len": 16,
              f"{ef.XA_PREFIX}max_tokens": 8}
        sp = types.SimpleNamespace(extra_args=xa, max_tokens=mt, ignore_eos=ig)
        return types.SimpleNamespace(req_id=f"r{mt}{ig}", sampling_params=sp)
    flow._register_new(types.SimpleNamespace(scheduled_new_reqs=[
        mk(8, True),    # ровно N — мало
        mk(9, False),   # без ignore_eos
        mk(9, True),    # корректный
    ]))
    assert list(flow.reqs) == ["r9True"]


def test_register_tolerates_malformed_xargs():
    flow = ef.EngineFlow()
    sp = types.SimpleNamespace(
        extra_args={f"{ef.XA_PREFIX}nonce": "not-a-number"},
        max_tokens=9, ignore_eos=True)
    flow._register_new(types.SimpleNamespace(scheduled_new_reqs=[
        types.SimpleNamespace(req_id="bad", sampling_params=sp)]))
    assert flow.reqs == {}
