# Residual engine seams for mixed PoC (vLLM 0.25)

`v0_25_mixed_seams.patch` — the full diff of the mixed-PoC vLLM branch
(axeltec-software/vllm `poc-decode-0.25`) against upstream v0.25.1.
Vanilla vLLM + this patch is exactly that branch.

Deployment: apply the patch (or use the branch), then
`pip install --no-deps` this package. Engine seams resolve the PoC
implementation through `vllm/poc/dispatch.py`: the gonka_poc plugin when
installed (production path), the in-tree copy otherwise (fallback). The
boot log states which one is active.

| seam | files | why it cannot live in the plugin |
| --- | --- | --- |
| scheduler admission hooks + async placeholder exemption | `v1/core/sched/scheduler.py`, `sched/output.py` | admission happens inside `schedule()`; PoC rows must be exempt from the async output-placeholder cap (they never produce sampled tokens) |
| request field plumbing | `v1/request.py`, `v1/engine/__init__.py`, `outputs.py`, `gpu_input_batch.py` | `poc_params`/`poc_outputs` ride engine types across processes |
| runner hooks (V1 + V2) | `gpu_model_runner.py`, `gpu/model_runner.py` | must fire before the cudagraph replay branch on every step; V2 also needs sampler-slot neutralization (`clear_slot`) |
| entry/exit | `async_llm.py`, `output_processor.py`, `api_server.py` | request creation (unique per-nonce cache salt) and emit-once artifact return |
| config knobs | `config/cache.py`, `config/vllm.py` | poc_share / route_window / KV-derived AUTO cap |

Invariants the seams encode (each covered by tests on the branch):
unique per-nonce cache salt (prefix caching must never share KV across
nonces), KV-derived AUTO batch cap (admitting beyond pool capacity
livelocks), emit-once guard (pipelined steps must not re-emit artifacts),
class-level module patching (torch.compile traces class forwards; module
replacement breaks the compiled parameter map), hooks before the
FULL-replay branch, async placeholder exemption for PoC rows.

Native transforms: one implementation, `mixed/native.py` — class-level
dispatch with in-graph seeded embeddings and sphere snap, the only shape
that works inside vLLM's compiled forward. Attachment is model-agnostic:
decoder layers by list, embedding/norm by attribute, MoE routers by
gate+experts discovery with a hard failure on unreadable expert metadata.
Grouped top-k routers (DeepSeek family, incl. Kimi) are forced with the
two-stage seeded formula (group salt + expert salt, coverage guarantee) —
consensus constants per the approved phase-0 spec; flat routers degenerate
to the one-stage scheme bit-exactly. Production route window: 256 (full
scatter).

Acceptance gates for any change to this stack: the cross-version golden
trajectory (block=PARITY, nonce=7, seq_len=64, max_tokens=8 ->
[1,15,13,14,0,13,4,5,1] on the reference model), the bit-parity suite
against the 0.20 sources, and measurement under async scheduling.
Sync-only results do not count as verification.
