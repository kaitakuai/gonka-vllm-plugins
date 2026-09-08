"""Scope vLLM's compile cache by the plugin code and knobs that shape the traced forward.

vLLM keys its torch.compile cache (compiled graphs and the AOT artifacts) by its own
environment variables, the engine config and the *source* of the files it traced
when it compiled. The plugin's in-graph transforms (class-level forward patches,
see mixed/native.py) are not among those files unless they were traced at that
compile, and the AOT artifact is loaded without tracing at all. So a host whose
cache was written by a build without them (3.0.16 hooks only) or with a different
version of them loads that graph and runs a forward without PoC ops: on 08.09.2026,
MiniMax on 1xB300, the decode scheme rode 3.0.16's cached graph — reflection,
seeded routing and the synthesized decode input all missing, chains independent
of the block hash beyond the prefill step. Knobs that change the traced forward
(ablation, debug prints) have the same effect.

The fix at the right layer: the plugin points ``VLLM_CACHE_ROOT`` at a
sub-directory named by a hash of its own package source plus the knob values, in
every process (vLLM loads general plugins in each one), before anything is
compiled. A cache written by another plugin build is never loaded.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)

_MARK = "gonka-poc-build-"

# knob -> default value (after normalisation). Extend when a new knob changes
# the traced forward; consensus parameters are NOT knobs (source constants).
_TRACED_KNOBS: dict[str, str] = {
    "POC_ABLATE": "",           # diagnostic ablation of PoC interventions
    "VLLM_POC_DEBUG_TP": "",    # debug prints inside the forward
}


def _normalise(name: str, raw: str | None) -> str:
    v = (raw or "").strip().lower()
    if name == "POC_ABLATE":
        return ",".join(sorted(x.strip() for x in v.split(",") if x.strip()))
    if name == "VLLM_POC_DEBUG_TP":
        return "1" if v == "1" else ""
    return v


def source_hash() -> str:
    """sha256 over every .py of the installed gonka_poc package (path + content)."""
    root = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            h.update(os.path.relpath(path, root).encode())
            with open(path, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def knob_signature(environ=None) -> tuple[dict[str, str], str]:
    """Return (normalised non-default knobs, signature over source + knobs)."""
    env = os.environ if environ is None else environ
    active = {k: _normalise(k, env.get(k)) for k in _TRACED_KNOBS}
    active = {k: v for k, v in active.items() if v != _TRACED_KNOBS[k]}
    payload = json.dumps({"source": source_hash(), "knobs": active}, sort_keys=True)
    return active, hashlib.sha256(payload.encode()).hexdigest()[:10]


def scope_compile_cache(environ=None) -> str:
    """Point ``VLLM_CACHE_ROOT`` at the sub-directory for this build; idempotent.

    Returns the scoped root."""
    env = os.environ if environ is None else environ
    active, sig = knob_signature(env)
    root = env.get("VLLM_CACHE_ROOT") or os.path.expanduser("~/.cache/vllm")
    root = root.rstrip("/")
    base, last = os.path.split(root)
    if last.startswith(_MARK):
        if last == _MARK + sig:
            return root
        root = base                       # a parent scoped for another build
    scoped = os.path.join(root, _MARK + sig)
    env["VLLM_CACHE_ROOT"] = scoped
    logger.info("PoC: compile cache scoped to %s (source %s, knobs %s)",
                scoped, sig, active)
    return scoped
