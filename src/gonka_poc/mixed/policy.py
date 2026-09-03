# SPDX-License-Identifier: Apache-2.0
"""Pure mixing-policy functions, ported VERBATIM from Ilya Slavutin's in-tree
port (axeltec-software/vllm @ poc-decode-0.25, vllm/poc/mixed_decode.py).
Kept pure/unit-testable exactly as authored; only the import home changed.
"""

import os

# Bound on consecutive chat-prefill defers before a decoding PoC is forced an
# exclusive step (fairness valve — keeps PoC from starving under chat churn).
POC_DEFER_LIMIT = 4


def poc_is_pure_path(poc_params) -> bool:
    """True for prefill-only PoC (max_tokens == 0), which has no decode loop. All
    decode — generation and validation — runs step-driven. Pure (unit-testable)."""
    return poc_params.max_tokens == 0


def poc_chat_like() -> bool:
    """PoC-строки планируются как чат: префилл делит шаг с декодом, без
    uniform-step-изоляции. Включено по умолчанию; POC_CHAT_LIKE=0 возвращает
    uniform-step. Обоснование и замеры — ADR-0016 §5."""
    v = os.environ.get("POC_CHAT_LIKE", "").strip().lower()
    return v not in ("0", "false", "no", "off")


def decode_only_mixing_gate(
    *,
    mixed_cudagraph: bool,
    poc_decode_pending: bool,
    poc_will_prefill: bool,
    chat_will_prefill: bool,
    consecutive_defers: int,
    defer_limit: int = POC_DEFER_LIMIT,
) -> tuple[bool, bool, int]:
    """Decide (defer_chat, defer_poc, consecutive_defers) so chat and PoC share a
    forward only when both decode; prefills run isolated. Mutually exclusive defers.
    Pure (unit-testable). With mixed_cudagraph=False reduces to the original
    behaviour (defer_chat=poc_decode_pending, defer_poc=False). The valve bounds
    consecutive chat-prefill defers so chat churn can't starve a decoding PoC.
    """
    defer_chat = poc_decode_pending or (mixed_cudagraph and poc_will_prefill)
    defer_poc = mixed_cudagraph and (not defer_chat) and chat_will_prefill
    if defer_poc:
        consecutive_defers += 1
        if consecutive_defers > defer_limit:
            # Give the decoding PoC one exclusive (pure-decode, graphable) step.
            defer_poc, defer_chat, consecutive_defers = False, True, 0
    else:
        consecutive_defers = 0
    return defer_chat, defer_poc, consecutive_defers


def poc_step_num_tokens(poc_params, num_computed_tokens: int) -> int:
    """Tokens to schedule for a PoC request this step: mixed decode generation
    prefills seq_len once then 1 token/step; the pure / prefill-only path is a
    single seq_len step. Pure (unit-testable)."""
    if not poc_is_pure_path(poc_params):
        return poc_params.seq_len if num_computed_tokens == 0 else 1
    return poc_params.seq_len


def poc_share_budget(poc_share: float, token_budget: int,
                     chat_present: bool = True) -> int:
    """PoC's slice of a step's compute (token) budget. poc_share=0 -> PoC blocked
    this step; 1.0 -> PoC may use the whole budget. Pure (unit-testable).

    The share exists to stop PoC starving chat, so it only applies while chat is
    actually in the engine: with no chat request queued or running, reserving a
    slice for it just idles the step (PoC-only nodes prefill nonces in twice the
    steps they need). poc_share=0 still blocks PoC either way — an explicit
    "chat only" instruction, not a reservation."""
    if not chat_present and poc_share > 0.0:
        return token_budget
    return int(poc_share * token_budget)


def poc_alloc_footprint(poc_params, num_new_tokens: int) -> int:
    """Dynamic-KV blocks to allocate: the pure path runs the whole decode loop in
    one step so it allocates seq_len+max_tokens upfront; the mixed path allocates
    one step's tokens. Pure (unit-testable)."""
    if poc_is_pure_path(poc_params):
        return poc_params.seq_len + poc_params.max_tokens
    return num_new_tokens


# PoC knobs live in our fork's CacheConfig. On a stock vLLM those attributes do
# not exist and the plugin must still run — that is the point of shipping it as a
# plugin — so every read goes through poc_cfg() and falls back to the SAME value
# the fork declares. If a default drifts, consensus-relevant behaviour (seq_len,
# max_tokens) would silently differ between a fork deploy and a stock deploy.
POC_CONFIG_DEFAULTS = {
    "poc_max_batch_size": 0,
    "poc_seq_len": 256,
    "poc_max_tokens": 256,
    "poc_share": 0.5,
    "poc_vector_artifacts": False,
}


def poc_cfg(cache_config, name: str):
    """Read a PoC knob from a CacheConfig that may not define it."""
    if name not in POC_CONFIG_DEFAULTS:
        raise KeyError(f"unknown PoC config knob: {name}")
    return getattr(cache_config, name, POC_CONFIG_DEFAULTS[name])
