# SPDX-License-Identifier: Apache-2.0
"""PoC configuration knobs shared by the mixed path.

The step policy that used to live here (mixing gate, token share, footprint)
is gone: PoC rows are scheduled by vLLM like chat (ADR-0017)."""

ADDITIONAL_CONFIG_KEY = "gonka_poc"


def poc_is_pure_path(poc_params) -> bool:
    """True for prefill-only PoC (max_tokens == 0), which has no decode loop. All
    decode — generation and validation — runs step-driven. Pure (unit-testable)."""
    return poc_params.max_tokens == 0


# PoC knobs ride in vLLM's public ``--additional-config`` under the "gonka_poc"
# key (``VllmConfig.additional_config``); the engine declares nothing for us.
# Every read goes through poc_cfg() and falls back to the SAME default on any
# tree. The consensus inputs (seq_len, max_tokens, k_dim, scheme) are not knobs:
# they arrive in every request from the chain. The batch is bounded by the
# engine's own ``--max-num-seqs`` and cudagraph capture size.
POC_CONFIG_DEFAULTS = {
    "poc_vector_artifacts": False,   # emit per-step vectors alongside the k trajectory
}


def poc_cfg(vllm_config, name: str):
    """Read a PoC knob from ``vllm_config.additional_config["gonka_poc"]``.

    Accepts ``None`` / any object without ``additional_config`` (tests, partial
    runners) and returns the default. Values are coerced to the default's type
    so ``--additional-config '{"gonka_poc": {"poc_vector_artifacts": "1"}}'``
    behaves like the boolean form.
    """
    if name not in POC_CONFIG_DEFAULTS:
        raise KeyError(f"unknown PoC config knob: {name}")
    default = POC_CONFIG_DEFAULTS[name]
    extra = getattr(vllm_config, "additional_config", None) or {}
    section = extra.get(ADDITIONAL_CONFIG_KEY) if isinstance(extra, dict) else None
    if not isinstance(section, dict) or name not in section:
        return default
    value = section[name]
    if isinstance(default, bool):
        return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")
    return type(default)(value)
