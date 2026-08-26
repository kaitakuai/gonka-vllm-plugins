# Engine seams for mixed PoC (vLLM 0.25)

`v0_25_mixed_seams.patch` — the diff of the mixed-PoC engine branch against
upstream v0.25.1. Stock vLLM plus this patch is exactly that branch.

The patch carries **only what cannot live in a plugin**: hook points inside
vLLM, plus the wire types that must be one class per process. The PoC
implementation itself is the `gonka_poc` package — there is no in-tree copy and
no fallback. Without the plugin installed the engine imports fail at startup,
which is the intended behaviour: a silently different implementation is worse
than a refusal to boot.

Deployment: apply the patch (or use the branch), then `pip install --no-deps`
the plugin. The consensus codebook ships inside the plugin package.

| seam | files | why it cannot live in the plugin |
| --- | --- | --- |
| scheduler admission + async placeholder exemption | `v1/core/sched/scheduler.py`, `sched/output.py` | admission happens inside `schedule()`; PoC rows must be exempt from the async output-placeholder cap (they never produce sampled tokens) |
| request field plumbing | `v1/request.py`, `v1/engine/__init__.py`, `outputs.py`, `v1/outputs.py`, `gpu_input_batch.py` | `poc_params` / `poc_outputs` ride engine types across processes |
| runner hooks (V1 + V2) | `gpu_model_runner.py`, `gpu/model_runner.py`, `gpu/sample/prompt_logprob.py` | must fire before the cudagraph replay branch on every step; V2 also needs sampler-slot neutralization |
| entry/exit | `async_llm.py`, `output_processor.py`, `api_server.py` | request creation (unique per-nonce cache salt) and emit-once artifact return |
| config knobs | `config/cache.py` | poc_share / route_window / KV-derived AUTO cap |
| custom ops | `_custom_ops.py` | op registration |
| wire types | `poc/poc_params.py`, `poc/__init__.py` | one class per process; cannot be resolved per-caller |

Invariants the seams encode, each covered by `tests/poc/unit/`: unique
per-nonce cache salt (prefix caching must never share KV across nonces),
KV-derived AUTO batch cap (admitting beyond pool capacity livelocks), emit-once
guard (pipelined steps must not re-emit artifacts), hooks before the FULL-replay
branch, async placeholder exemption for PoC rows.

Acceptance gates for any change to this stack: the cross-version golden
trajectory (block=PARITY, nonce=7, seq_len=64, max_tokens=8 →
[1,15,13,14,0,13,4,5,1] on the reference model), the bit-parity suite against
the 0.20 sources, and measurement under async scheduling. Sync-only results do
not count as verification.
