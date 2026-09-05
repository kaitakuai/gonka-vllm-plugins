# SPDX-License-Identifier: Apache-2.0
"""PoC configuration knobs shared by the mixed path.

The step policy that used to live here (mixing gate, token share, footprint)
is gone: PoC rows are scheduled by vLLM like chat (ADR-0017)."""


def poc_is_pure_path(poc_params) -> bool:
    """True for prefill-only PoC (max_tokens == 0), which has no decode loop. All
    decode — generation and validation — runs step-driven. Pure (unit-testable)."""
    return poc_params.max_tokens == 0


# PoC knobs live in our fork's CacheConfig. On a stock vLLM those attributes do
# not exist and the plugin must still run — that is the point of shipping it as a
# plugin — so every read goes through poc_cfg() and falls back to the SAME value
# the fork declares. If a default drifts, consensus-relevant behaviour (seq_len,
# max_tokens) would silently differ between a fork deploy and a stock deploy.
POC_CONFIG_DEFAULTS = {
    # decode-state slots: 0 = max_num_seqs (vLLM never runs more rows than that)
    "poc_max_batch_size": 0,
    "poc_seq_len": 256,
    "poc_max_tokens": 256,
    "poc_vector_artifacts": False,
}


def poc_cfg(cache_config, name: str):
    """Read a PoC knob from a CacheConfig that may not define it."""
    if name not in POC_CONFIG_DEFAULTS:
        raise KeyError(f"unknown PoC config knob: {name}")
    return getattr(cache_config, name, POC_CONFIG_DEFAULTS[name])
