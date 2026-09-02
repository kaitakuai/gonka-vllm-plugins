# gonka-poc

vLLM plugin implementing **Gonka Proof-of-Compute** with mixed execution:
PoC nonces ride the serving pipeline as engine requests — same scheduler,
same KV cache manager, same compiled CUDA graphs as chat traffic. The proof
therefore certifies the kernels that serve real users.

## Architecture

Two byte-identical homes for the implementation:

* **This plugin** (`gonka_poc`) — the production path.
* **In-tree copy** on the vLLM branch — the standalone fallback.

The engine resolves between them at import time (`vllm/poc/dispatch.py` on
the branch): plugin when installed, in-tree otherwise. The boot log states
which is active.

The engine itself carries a small set of committed seams (scheduler
admission, request-field plumbing, runner hooks, entry/exit, config knobs)
— no runtime monkeypatching. `residual/README.md` documents each seam and
why it cannot live in the plugin; `residual/v0_25_mixed_seams.patch` is the
full diff against upstream vLLM v0.25.1.

## Deployment

```
pip install vllm==0.25.1          # or use the patched branch directly
git apply residual/v0_25_mixed_seams.patch    # skip if using the branch
pip install --no-deps -e .        # this package
vllm serve <model>
```

Boot log confirms: `PoC implementation: gonka_poc... (plugin)`.
PoC API: `POST /api/v1/pow/generate` (see `src/gonka_poc/poc/routes.py`).

## Runtime knobs (plugin-side, environment of the server)

| variable | default | meaning |
| --- | --- | --- |
| `POC_CHAT_LIKE` | off | PoC rows are scheduled like chat: prefill shares the step with decode, no uniform-step isolation. Faster alone (~5%), worse next to live chat. |
| `POC_KV_HEADROOM` | `0.01` | fraction of the KV pool kept free when admitting PoC prefills (plus one block per running row). |
| `POC_FUSED_REFLECT` | `1` | Triton one-pass Householder reflection; `0` restores the four-kernel reference path. |
| `POC_ROLLING_WINDOW` / `POC_ROLLING_WAVE` | off | client-side rolling admission (window of concurrent nonces, wave size). |
| `POC_PREFILL_PER_STEP` | `0` | at most k new PoC prefills per step (meaningful with `POC_CHAT_LIKE`). |
| `POC_DIAG` | off | step-interval histograms with composition, stall/alloc pool dumps, phase timers (diagnostics only). |
| `POC_ABLATE` | off | `reflect,router,pseudo` — disable PoC interventions for diagnosis; not a consensus mode. |

`poc_max_batch_size`, `poc_share`, `poc_seq_len`, `poc_max_tokens` are read
through `poc_cfg()` with the plugin's own defaults; the residual carries no CLI
arguments for PoC. See [ADR-0016](docs/adr/ADR-0016-hopper-admission-and-fused-reflection.md).

## Package layout

```
src/gonka_poc/
  poc/          consensus core + API: seeds/inputs (gpu_random), sphere
                codebook + snap, artifacts + fraud test (data, validation),
                routes, generate_queue, callbacks
  mixed/        engine-facing seam counterparts: admission (scheduler),
                bridge (model runners, V1 + V2), runtime (decode chain +
                emit-once), native (in-graph transforms: seeded embeddings,
                Householder reflections, seeded MoE routing, sphere snap;
                class-level dispatch — survives torch.compile)
  entrypoint/   serve wrapper + gating middleware
  worker/       worker-extension base
  plugin.py     vllm.general_plugins entry point
residual/       engine seam patch + seam documentation
```

## Invariants

* PoC produces no token and never enters the sampler/output path.
* Async scheduling is the production configuration; sync-only results do
  not count as verification.
* All PoC forwards run inside captured CUDA graphs; transforms, routing
  and snap are in-graph.
* Unique per-nonce cache salt; pure dynamic KV; chat KV untouched.
* Seeded MoE routing is part of the algorithm, not a toggle; router
  discovery hard-fails on unreadable expert metadata.

## Tests

```
pytest tests/contract tests/unit             # CPU-only
DECODE020_PATH=<0.20 poc sources> pytest tests/unit   # + bit-parity suite
```

Live golden trajectories (dense + MoE) live on the vLLM branch under
`tests/poc/integration/` — any change there is a consensus change.
