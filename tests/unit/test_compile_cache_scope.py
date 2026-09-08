"""The compile cache is scoped by the plugin source and the knobs that change the traced forward."""
from gonka_poc import compile_cache
from gonka_poc.compile_cache import knob_signature, scope_compile_cache


def test_defaults_still_get_a_build_scoped_root():
    env = {"VLLM_CACHE_ROOT": "/tmp/vllm_r"}
    scoped = scope_compile_cache(env)
    assert scoped.startswith("/tmp/vllm_r/gonka-poc-build-")
    assert env["VLLM_CACHE_ROOT"] == scoped
    env2 = {"VLLM_CACHE_ROOT": "/tmp/vllm_r", "POC_ABLATE": "", "VLLM_POC_DEBUG_TP": "0"}
    assert scope_compile_cache(env2) == scoped          # explicit defaults == unset
    assert scope_compile_cache(env) == scoped           # child process inherits: unchanged


def test_source_change_moves_the_root(monkeypatch):
    env = {"VLLM_CACHE_ROOT": "/tmp/vllm_r"}
    scoped = scope_compile_cache(env)
    monkeypatch.setattr(compile_cache, "source_hash", lambda: "other-build")
    env = {"VLLM_CACHE_ROOT": "/tmp/vllm_r"}
    assert scope_compile_cache(env) != scoped


def test_non_default_knob_gets_its_own_root():
    base = scope_compile_cache({"VLLM_CACHE_ROOT": "/tmp/vllm_r"})
    env = {"VLLM_CACHE_ROOT": "/tmp/vllm_r", "VLLM_POC_DEBUG_TP": "1"}
    scoped = scope_compile_cache(env)
    assert scoped != base and scoped.startswith("/tmp/vllm_r/gonka-poc-build-")
    env2 = {"VLLM_CACHE_ROOT": "/tmp/vllm_r", "VLLM_POC_DEBUG_TP": "1", "POC_ABLATE": "router,reflect"}
    assert scope_compile_cache(env2) != scoped
    env3 = {"VLLM_CACHE_ROOT": "/tmp/vllm_r", "POC_ABLATE": "reflect, router"}
    assert knob_signature(env3)[0] == {"POC_ABLATE": "reflect,router"}   # order-insensitive


def test_parent_scoped_for_another_build_is_replaced():
    env = {"VLLM_CACHE_ROOT": "/tmp/vllm_r/gonka-poc-build-deadbeef00"}
    scoped = scope_compile_cache(env)
    assert scoped.startswith("/tmp/vllm_r/gonka-poc-build-") and "deadbeef00" not in scoped
