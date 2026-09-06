#!/usr/bin/env python3
"""Parity of prefill-scheme vectors between two runs (e.g. eager vs compiled).

    python tools/prefill_compare.py a.artifacts.json b.artifacts.json [tol=0.01]

Prints the cosine-distance distribution over common nonces and how many exceed
the validator's per-vector tolerance (VLLM_POC_VECTOR_TOL, default 0.01).
"""
import base64
import json
import sys

import numpy as np


def load(path):
    d = json.load(open(path))
    return {n: np.frombuffer(base64.b64decode(v), dtype="<f2").astype(np.float64)
            for n, v in d.items() if v}


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    common = sorted(set(a) & set(b), key=int)
    if not common:
        sys.exit("no common nonces")
    dists = []
    for n in common:
        va, vb = a[n], b[n]
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        dists.append(1.0 - float(va @ vb) / (na * nb) if na and nb else 1.0)
    d = np.array(dists)
    print(f"common nonces: {len(common)}; cosine distance: max {d.max():.2e}, "
          f"mean {d.mean():.2e}, median {np.median(d):.2e}, p99 {np.percentile(d, 99):.2e}")
    print(f"exact (dist == 0): {(d == 0).sum()}; over tol {tol}: {(d > tol).sum()}; "
          f"over 0.02: {(d > 0.02).sum()}")


if __name__ == "__main__":
    main()
