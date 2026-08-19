# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the mixed subpackage.

1. One-way dependency: no gonka_poc core module imports gonka_poc.mixed.
2. The pre-forward hook is registrable, fires with the residual-contract
   signature, never raises, and records evidence.
"""
import pathlib
import types

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "gonka_poc"


def test_core_never_imports_mixed():
    """AST-разбор настоящих импортов — комментарии и строки не триггерят."""
    import ast
    offenders = []
    for p in SRC.rglob("*.py"):
        if "mixed" in p.parts:
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                names = [mod] + [f"{mod}.{a.name}" for a in node.names]
            else:
                continue
            if any(n == "gonka_poc.mixed" or n.startswith("gonka_poc.mixed.")
                   or n == "mixed" or n.endswith(".mixed") for n in names):
                offenders.append(f"{p.relative_to(SRC)}: {names}")
    assert not offenders, f"core imports mixed (one-way rule): {offenders}"


def test_hook_installs_and_fires(monkeypatch):
    from gonka_poc.mixed import pre_forward

    runner = types.SimpleNamespace(pre_forward_hooks=[])
    monkeypatch.setenv("POC_MIXED_PRE_FORWARD", "1")
    assert pre_forward.install(runner) is True
    assert pre_forward.install(runner) is True  # idempotent
    assert runner.pre_forward_hooks.count(pre_forward.poc_pre_forward) == 1

    sched = types.SimpleNamespace(total_num_scheduled_tokens=536,
                                  num_scheduled_tokens={"r1": 1, "r2": 1})
    before = pre_forward.stats()["calls"]
    for hook in runner.pre_forward_hooks:
        hook(runner, sched, None, None, None, object())
    s = pre_forward.stats()
    assert s["calls"] == before + 1
    assert s["last_num_tokens"] == 536
    assert s["last_num_reqs"] == 2
    assert s["has_attn_metadata"] is True
    assert s["errors"] == 0


def test_hook_never_raises():
    """Скелет observe-only обязан глотать СВОИ ошибки (не тавтология: подаём
    scheduler_output, который ломает внутреннюю арифметику)."""
    from gonka_poc.mixed import pre_forward
    before = pre_forward.stats()["errors"]
    bad = types.SimpleNamespace(total_num_scheduled_tokens="abc",
                                num_scheduled_tokens=None)
    pre_forward.poc_pre_forward(None, bad, None, None, None, None)
    s = pre_forward.stats()
    assert s["errors"] == before + 1  # ошибка учтена, наружу не вышла


def test_hook_signature_matches_residual_seam():
    """Пин сигнатуры к резидуальному колл-сайту (kaitakuai/vllm@13e6bacd):
    _hook(self, scheduler_output, input_ids, positions, inputs_embeds,
    attn_metadata). Дрейф арности = TypeError на КАЖДОМ шаге движка."""
    import inspect
    from gonka_poc.mixed import pre_forward
    params = list(inspect.signature(pre_forward.poc_pre_forward).parameters)
    assert params == ["runner", "scheduler_output", "input_ids", "positions",
                      "inputs_embeds", "attn_metadata"]


def test_hook_records_ubatched_shape():
    from gonka_poc.mixed import pre_forward
    sched = types.SimpleNamespace(total_num_scheduled_tokens=8,
                                  num_scheduled_tokens={"r": 8})
    pre_forward.poc_pre_forward(None, sched, None, None, None, [{}, {}])
    assert pre_forward.stats()["attn_is_ubatched"] is True
    pre_forward.poc_pre_forward(None, sched, None, None, None, {})
    assert pre_forward.stats()["attn_is_ubatched"] is False


def test_install_refuses_without_residual_seam(monkeypatch):
    from gonka_poc.mixed import pre_forward
    monkeypatch.setenv("POC_MIXED_PRE_FORWARD", "1")
    with pytest.raises(RuntimeError, match="pre_forward_hooks"):
        pre_forward.install(types.SimpleNamespace())


def test_policy_pure_functions():
    from gonka_poc.mixed import policy
    # клапан справедливости: после лимита отложек PoC получает эксклюзивный шаг
    d = 0
    for _ in range(policy.POC_DEFER_LIMIT):
        defer_chat, defer_poc, d = policy.decode_only_mixing_gate(
            mixed_cudagraph=True, poc_decode_pending=False,
            poc_will_prefill=False, chat_will_prefill=True,
            consecutive_defers=d)
        assert defer_poc and not defer_chat
    defer_chat, defer_poc, d = policy.decode_only_mixing_gate(
        mixed_cudagraph=True, poc_decode_pending=False,
        poc_will_prefill=False, chat_will_prefill=True, consecutive_defers=d)
    assert defer_chat and not defer_poc and d == 0
    p = types.SimpleNamespace(seq_len=256, max_tokens=256)
    assert policy.poc_step_num_tokens(p, 0) == 256
    assert policy.poc_step_num_tokens(p, 256) == 1
    assert policy.resolve_poc_max_batch_size(0, 704) == 704
    assert policy.resolve_poc_max_batch_size(536, 704) == 536
