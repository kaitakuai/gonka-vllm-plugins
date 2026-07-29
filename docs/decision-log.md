# Decision log

Short, factual, link-rich. One entry per decision that outlives the PR that
made it. Full rationale lives in `docs/adr/`.

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
