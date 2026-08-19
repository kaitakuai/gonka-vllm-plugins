# SPDX-License-Identifier: Apache-2.0
"""Worker extension for mixed mode: the core PoCWorkerExtension plus the
mixed seam controls. Selected at launch instead of the core class:

    --worker-extension-cls gonka_poc.mixed.extension.MixedPoCWorkerExtension

Core stays untouched (one-way dependency: mixed imports core, never the
reverse) and a node launched with the core class physically cannot enter
mixed mode.
"""
from gonka_poc.worker.extension import PoCWorkerExtension

from . import pre_forward


class MixedPoCWorkerExtension(PoCWorkerExtension):
    """PoCWorkerExtension + mixed-mode RPC surface (methods become attributes
    on the live GPU Worker; keep the ``mixed_``/``execute_poc_`` prefixes
    unique — vLLM asserts no collisions at init_worker time)."""

    def mixed_enable_pre_forward(self) -> dict:
        """Register the pre-forward hook on this worker's runner (idempotent).
        Requires the residual with pre_forward_hooks AND POC_MIXED_PRE_FORWARD=1."""
        installed = pre_forward.install(self.model_runner)
        return {"rank": getattr(self, "rank", -1), "installed": bool(installed)}

    def mixed_hook_stats(self) -> dict:
        """Drain smoke evidence: hook call count + last step context summary."""
        out = pre_forward.stats()
        out["rank"] = getattr(self, "rank", -1)
        return out

    def mixed_enable_engine_flow(self) -> dict:
        """Register the PoC-as-a-request hooks (pre: row layout, post: k-snap)
        on this worker's runner. Requires the residual with BOTH seams
        (branch mixed/poc-as-request) and POC_ENGINE_FLOW=1."""
        from . import engine_flow
        ok = engine_flow.install(self.model_runner)
        return {"rank": getattr(self, "rank", -1), "installed": bool(ok)}

    def mixed_collect_artifacts(self) -> dict:
        """Drain finished engine-flow artifacts (k-id chains) accumulated on
        this worker. Msgpack-safe: ints/lists only."""
        from . import engine_flow
        out = engine_flow.FLOW.collect(self.model_runner)
        out["rank"] = getattr(self, "rank", -1)
        return out
