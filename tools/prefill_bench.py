#!/usr/bin/env python3
"""Throughput of the prefill PoC scheme (v0.1.x artifacts) against a live server.

Posts /api/v1/pow/generate with ``scheme="prefill"`` and ``wait=true`` for N nonces
at each batch size, samples GPU utilisation meanwhile, and prints nonce/s.

    POC_URL=http://127.0.0.1:8000 MODEL=MiniMaxAI/MiniMax-M2.7 \
    BATCHES=32,64,128 NONCES=512 REPEATS=3 python tools/prefill_bench.py out.jsonl

``batch_size`` is what the node passes per round: the server splits the nonce
list into forwards of that many nonces (poc_model_runner, one forward per chunk).
"""
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time

import requests

URL = os.environ.get("POC_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("MODEL", "MiniMaxAI/MiniMax-M2.7")
SEQ_LEN = int(os.environ.get("SEQ_LEN", "256"))
BATCHES = [int(x) for x in os.environ.get("BATCHES", "32,64,128").split(",")]
NONCES = int(os.environ.get("NONCES", "512"))
REPEATS = int(os.environ.get("REPEATS", "3"))
BLOCK_HASH = hashlib.sha256(b"prefill-bench").hexdigest()
PUBLIC_KEY = hashlib.sha256(b"prefill-bench-pk").hexdigest()[:40]


class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"], text=True, timeout=5)
                self.samples.append(float(out.split("\n")[0].strip()))
            except Exception:
                pass
            self._stop.wait(0.5)

    def stop(self):
        self._stop.set()
        self.join(timeout=5)
        return statistics.mean(self.samples) if self.samples else float("nan")


def generate(nonces, batch_size):
    body = {
        "block_hash": BLOCK_HASH, "block_height": 1, "public_key": PUBLIC_KEY,
        "node_id": 0, "node_count": 1, "nonces": list(nonces),
        "batch_size": batch_size, "wait": True,
        "params": {"model": MODEL, "seq_len": SEQ_LEN, "k_dim": 12,
                   "max_tokens": 0, "scheme": "prefill"},
    }
    t0 = time.perf_counter()
    r = requests.post(f"{URL}/api/v1/pow/generate", json=body, timeout=3600)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    arts = data.get("artifacts") or data.get("results") or []
    return len(arts), dt, data


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "prefill_bench.jsonl"
    # warm-up: first forward of a fresh boot pays JIT/allocation costs
    n, dt, _ = generate(range(BATCHES[0]), BATCHES[0])
    print(f"warm-up: {n} artifacts in {dt:.2f} s", flush=True)
    rows = []
    with open(out_path, "a") as fh:
        for b in BATCHES:
            rates, utils = [], []
            for rep in range(REPEATS):
                base = 1_000_000 * (BATCHES.index(b) + 1) + rep * NONCES
                gpu = GpuSampler(); gpu.start()
                n, dt, data = generate(range(base, base + NONCES), b)
                util = gpu.stop()
                rate = n / dt if dt else 0.0
                rates.append(rate); utils.append(util)
                rec = {"batch": b, "nonces": NONCES, "artifacts": n, "sec": round(dt, 3),
                       "nonce_per_s": round(rate, 2), "gpu_util": round(util, 1),
                       "model": MODEL, "url": URL}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"batch {b:4d}: {n:4d} artifacts in {dt:7.2f} s = {rate:7.2f} nonce/s, GPU {util:5.1f} %",
                      flush=True)
            rows.append((b, statistics.median(rates), statistics.mean(utils)))
    print("\n| batch | nonce/s (median) | GPU % |")
    print("| ---: | ---: | ---: |")
    for b, r, u in rows:
        print(f"| {b} | {r:.1f} | {u:.0f} |")


if __name__ == "__main__":
    main()
