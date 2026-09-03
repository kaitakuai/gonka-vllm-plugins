# Decision log

Short, factual, link-rich. One entry per decision that outlives the PR that
made it. Full rationale lives in `docs/adr/`.

## 2026-09-02 — Hopper decode-PoC: admission defects fixed, fused reflection

R on 4×H100 (DeepSeek-V4, vLLM 0.28) went from 0.47 to 1.07 without touching
kernels, TP or the PoC math. The hang above ~160 nonces was a livelock of the
uniform-step rule; the low R was a per-step cap of 134 rows computed from a KV
formula that misreads hybrid KV block sizes. Six admission defects fixed, one
Triton kernel added. Chat-like scheduling became the default on 03.09 (the
earlier uniform-step advantage next to live chat came from an admission scan
that bypassed the isolation); `POC_CHAT_LIKE=0` restores uniform-step. Verdict unchanged in every
corpus↔engine combination (0 at τ=0.05).

See [ADR-0016](adr/ADR-0016-hopper-admission-and-fused-reflection.md).

## 2026-07-24 — First tag cut: `v0.1.0a0`

The repository had no tags, so every downstream consumer pinned either a branch
or a loose commit. Tagged `2833a57c` as `v0.1.0a0` to give builds an immutable
anchor, and repointed the image builds at the tag.

Matters because the package is being handed over to another organisation: a
branch pin would silently start resolving to a different owner's HEAD.

## 2026-07-20 — Validation runs on leased KV blocks

PoC validation no longer aborts live inference: it borrows KV blocks where the
configuration is provably scratch-free, and falls back to the abort-based path
everywhere else. The legacy derivation path is preserved unchanged on every config,
so consensus output is unaffected.

See [ADR-0015](adr/ADR-0015-poc-validation-kv-borrowing.md).

## 2026-06-16 — Plugin + thin residual, not a fork

PoC ships as two artifacts: everything reachable through vLLM's public
extension points lives in this plugin; only the sampler surfaces with no public
hook stay as an in-tree residual. Upstreaming the residual is deferred with no
owner assigned, so the residual is treated as permanent infrastructure rather
than a temporary bridge.

See [ADR-0014](adr/ADR-0014-plugin-vs-fork.md), [ADR-0013](adr/ADR-0013-poc-gate-ordering.md),
and `MIGRATION_FROM_FORK.md` for the per-commit port record.
