# Decision log

Short, factual, link-rich. One entry per decision that outlives the PR that
made it. Full rationale lives in `docs/adr/`.

## 2026-09-03 — vLLM 0.25.1 is the release line; Hopper TP>1 needs `--no-async-scheduling`

The release targets vLLM 0.25.1 only (Vlad's request), so the branch pair
`mixed-poc-vllm-0.25.1-dev` (residual 4ebecc816 + this plugin) was verified on it.
1×B300: PoC 27–28 nonce/s at GPU 92–95%, chat 25.3 req/s (c=512), R 1.08; corpora
produced on 0.28 validate on 0.25.1 with 0 mismatches at τ=0.05; REAP fraud arm
3.4% vs honest background ≤0.02%; NVFP4 boots (FLASHINFER_TRTLLM) but its prover is
non-deterministic at ~0.01%. Mixed batches stay the default (decode-only steps cut
chat from 13.4 to 5.0 req/s next to PoC).

4×H100 (TP=4) on 0.25.1 crashes the engine (`illegal instruction`) once a PoC
batch exceeds ~64 nonces, regardless of every plugin knob (mixed/decode-only,
fused reflect, prefill slicing, ablated math), CUDA graph mode, stream and
collective settings. Chat of the same shape does not crash; 0.28 does not crash.
Root cause not located. Deploy Hopper TP>1 nodes on 0.25.1 with
`--no-async-scheduling` (vLLM flag, node config): PoC 20.4 nonce/s at GPU 76%,
chat 21.9 req/s, R 0.93 (vs 27.9 / 28.3 / 0.99 on 0.28 with async scheduling);
cross-version and self validation stay at 0 mismatches (τ=0.05), but PoC next to
live chat degrades most (PoC 1500 in 151 s with chat at 4.7 req/s, vs 102 s and
8.3 req/s on 0.28).

## 2026-09-02 — Hopper decode-PoC: admission defects fixed, fused reflection

R on 4×H100 (DeepSeek-V4, vLLM 0.28) went from 0.47 to 1.07 without touching
kernels, TP or the PoC math. The hang above ~160 nonces was a livelock of the
decode-only-step rule; the low R was a per-step cap of 134 rows computed from a KV
formula that misreads hybrid KV block sizes. Six admission defects fixed, one
Triton kernel added. Mixed batches became the default on 03.09 (the
earlier decode-only-step advantage next to live chat came from an admission scan
that bypassed the isolation); `POC_MIXED_BATCH=0` (renamed from `POC_CHAT_LIKE` on 2026-09-03) restores decode-only steps. Verdict unchanged in every
corpus↔engine combination (0 at τ=0.05).

Pseudo token ids for hash-MoE stay and are always on; token-id-routed models
remain an explicit allow-list with a loud refusal for unknown ones (03.09).

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
