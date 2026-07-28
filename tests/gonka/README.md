# Gonka Integration Tests

Tests for Gonka-specific features exposed by the `gonka-poc` plugin: Proof
of Computation (PoC) generation/validation, inference validation with
enforced tokens, chat priority gating, grammar graceful degradation, and
`logprobs_mode` behaviour.

---

## Test Files

### Unit Tests (no server required)

These live in `tests/unit/`. None of them need a GPU or a running vLLM
server. CPU-only, fast, run on every CI bump. See `tests/unit/` for the
current set of files — each module docstring says what it covers.

There are also vLLM private-surface drift detectors at
`tests/contract/test_api_surface.py` — pins the upstream
symbols the `gonka_poc._compat.v0_25` shim depends on (class
names, dataclass fields, method signatures). Run on every vLLM pin bump.

### Live Tests (require running `gonka-vllm-serve`)

| File | What it tests |
|------|---------------|
| `test_live_chat_priority.py` | PoC activates → chat rejected 503 → PoC stops → chat resumes; long inference aborted by PoC, engine survives. |
| `test_live_grammar_degradation.py` | Structured output + enforced tokens replay; corrupted tokens don't crash engine; distance2 with grammar. |
| `test_live_inference.py` | Basic chat, logprobs, structured output, temperature, seed determinism, max_tokens, top_logprobs. |
| `test_live_validation.py` | Enforced token replay across params (temperature, seed, grammar, prompt lengths); corrupted tokens; text match; distance2. |
| `test_live_poc.py` | PoC artifact generation, self-validation L2 < 0.2, batch generation, different block hashes, server-side validation. |
| `test_live_logprobs_mode.py` | `processed_logprobs` vs `raw_logprobs` behaviour; -9999 clamping; per-request override; validation distance with matching/mismatched modes. |

### Shared Helpers

| File | Role |
|------|------|
| `live_conftest.py` | Shared helpers for live tests: `chat_request`, `build_enforced_tokens`, `extract_result`, `token_distance2`, `distance2` (matching the production validation pipeline). |

---

## Prerequisites

### Unit Tests + Contract Tests

CPU-only, no server needed. See [CONTRIBUTING.md](../../CONTRIBUTING.md).


### Live Tests

A running `gonka-vllm-serve` with a loaded model. The tests read
`VLLM_TEST_MODEL` and `VLLM_TEST_PORT` env vars, defaulting to
`Qwen/Qwen2.5-0.5B-Instruct` on port `18199`.

#### Quick start with Docker (small model)

```bash
docker run -d --rm \
  --gpus '"device=0"' \
  --name gonka-test \
  -p 18199:18199 \
  --shm-size=4g \
  -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  vllm/vllm-openai:v0.25.1 \
  sh -c "pip install 'git+https://github.com/kaitakuai/gonka-poc@main' && \
         gonka-vllm-serve \
           --model Qwen/Qwen2.5-0.5B-Instruct \
           --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
           --attention-backend FLASHINFER \
           --logprobs-mode processed_logprobs \
           --enforce-eager \
           --dtype float16 \
           --host 0.0.0.0 \
           --port 18199 \
           --max-model-len 4096 \
           --gpu-memory-utilization 0.4"

# Wait for server to be ready
while ! curl -s http://127.0.0.1:18199/health; do sleep 2; done
```

`--worker-extension-cls` is required — see the top-level README for why
`gonka-vllm-serve` does NOT auto-inject it.

#### Running against Qwen3-235B-A22B (FP8)

The same tests run against the production-scale MoE model. This validates
PoC, inference, and validation on the actual deployment target. Requires
4× A100-80GB (or H100/H200/B200) and the model pre-downloaded in the HF
cache.

```bash
docker run -d \
  --gpus '"device=4,5,6,7"' \
  --ipc=host \
  --name gonka-235b-test \
  -p 18200:18200 \
  --shm-size=16g \
  -e VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v /path/to/huggingface/cache:/root/.cache/huggingface \
  vllm/vllm-openai:v0.25.1 \
  sh -c "pip install 'git+https://github.com/kaitakuai/gonka-poc@main' && \
         gonka-vllm-serve \
           --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
           --worker-extension-cls gonka_poc.worker.PoCWorkerExtension \
           --attention-backend FLASHINFER \
           --logprobs-mode processed_logprobs \
           --enforce-eager \
           --dtype auto \
           --host 0.0.0.0 \
           --port 18200 \
           --tensor-parallel-size 4 \
           --max-model-len 4096 \
           --served-model-name Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

while ! curl -s http://127.0.0.1:18200/health; do sleep 5; done
```

> **A100 FP8 note:** A100 GPUs lack native FP8 compute, so vLLM uses
> Marlin weight-only FP8 decompression. This works correctly but is
> slower than native FP8 on H100/H200/B200.

Run the live suite against the 235B server:

```bash
VLLM_TEST_MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8" \
VLLM_TEST_PORT=18200 \
pytest tests/gonka/test_live_*.py -v -s
```

#### Run live tests against a smaller default server

```bash
pytest tests/gonka/test_live_*.py -v -s
```

If the server is not running, live tests are **automatically skipped**
(not failed).

#### Run specific test suites

```bash
# Only inference + validation
pytest tests/gonka/test_live_inference.py tests/gonka/test_live_validation.py -v -s

# Only PoC
pytest tests/gonka/test_live_poc.py -v -s

# Only chat priority gating
pytest tests/gonka/test_live_chat_priority.py -v -s
```

Tests can be run in **any order**. Each test file cleans up after itself
(PoC is stopped before and after every test via fixtures).

---

## Distance Metrics

The validation tests use the same `distance2` metric as the production
validation pipeline:

- **`token_distance2`**: Per-position normalized logprob distance. Iterates
  validation-side tokens, builds fallback from inference-side sorted
  logprobs.
- **`distance2`**: Sequence-level mean of `token_distance2` over all
  positions, normalized by `max(100, n_positions) * n_logprobs`.

Expected values for honest self-validation (same server, same model):

- **distance2 < 0.05** for all inference validation (with or without
  grammar).

For PoC self-validation:

- **L2 distance < 0.2** for individual pairs (test_02).
- **Mean L2 < 0.1, max L2 < 0.3** across 20 pairs (test_06 — wider max to
  avoid flakes from float16 variance).
