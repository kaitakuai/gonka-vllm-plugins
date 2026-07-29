# ADR-0013 — PoC gate / abort / spawn ordering contract (local stub)

**Status:** Accepted. Originated as mlnode-foundry ADR-0013 (kaitakuai
internal); this file is authoritative for this repo.
This file is a 1-page summary so an offline reader of the `gonka-poc` repo
can find the rationale for the inline comment in `init_generate` in
`src/gonka_poc/poc/routes.py` without leaving the package.

**Date:** 2026-06-12 (canonical); local stub 2026-06-17.
**Owners:** @baychak
**Supersedes / amends:** none. **Amended by:** [ADR-0014](ADR-0014-plugin-vs-fork.md).

## Decision (the part the plugin code depends on)

PoC `init/generate` MUST execute three steps in this exact order:

1. **`gate.activate("init-generate")`** — flips the `PoCGate` shared state.
   The Starlette `PoCGatingMiddleware` starts returning HTTP 503 with
   `Retry-After` to `/v1/chat/completions` and `/v1/completions` immediately.
   No module-level flag; the gate is the single source of truth for
   "PoC is currently running".
2. **`await compat.abort_all_requests(engine_client)`** — drains the set of
   already-admitted chat/completions requests that snuck in before the gate
   flipped. The middleware blocks NEW admissions; this call kills in-flight
   ones so the PoC forward runs on an exclusively-owned GPU.
3. **`asyncio.create_task(_generation_loop(...))`** — spawns the PoC
   generation task. Only after (1)+(2) have completed.

Inverting any pair breaks the contract:
- spawn-before-abort → PoC forward fights in-flight chat batches for KV
  cache and execution slots; the derived artifacts drift away from what
  other validators compute.
- spawn-before-activate → new chat requests stream in alongside PoC; same
  failure mode plus a 503-window race.
- activate-after-abort → the gap between `abort_all_requests` and
  `activate` admits new chat requests that the abort already missed.

`gen_task.add_done_callback` deactivates the gate when generation finishes
(or is cancelled) — see `_on_generation_done` in the same function.

## Dependencies

- `gonka_poc.entrypoint.gating.PoCGate` — the shared activate/deactivate
  flag (see `tests/unit/test_gating.py` for the Starlette contract).
- `gonka_poc._compat.current()` dispatch — `abort_all_requests` is private
  vLLM surface that varies across minors; the compat shim isolates the
  version-specific call site.

## Links

- Source citing this ADR: `init_generate` in `src/gonka_poc/poc/routes.py`
- Companion ADR: [ADR-0014](ADR-0014-plugin-vs-fork.md) (residual fork as
  permanent infrastructure)
