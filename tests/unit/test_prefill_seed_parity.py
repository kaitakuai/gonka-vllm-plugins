# SPDX-License-Identifier: Apache-2.0
"""The prefill scheme derives exactly as the shipped MLNode image does.

Not by matching a formula, but by construction: ``gpu_random`` IS the 0.1.3
file, and the decode scheme keeps its own salted draws in ``decode_random``.
These guard that split, because every regression we chased came from decode
reaching into the prefill file -- a flag, a branch, a different allocation --
each of which "kept the values identical" and still moved the artifact.
"""
import inspect

import pytest
import torch

from gonka_poc.poc import gpu_random as gr
from gonka_poc.poc import decode_random as dr

BH, PK, DEV = "block-parity", "pk-parity", torch.device("cpu")


def test_prefill_pick_takes_no_scheme_flag():
    """A scheme argument here is how decode leaks into the prefill derivation."""
    params = set(inspect.signature(gr.random_pick_indices).parameters)
    assert params == {"block_hash", "public_key", "nonces", "dim", "k", "device"}


@pytest.mark.parametrize("nonce", [0, 7, 4242])
def test_prefill_pick_carries_no_decode_salt(nonce):
    """The seed string is the 0.1.3 one: block, key, nonce, k -- and nothing else."""
    seed = gr._seed_from_string(f"{BH}_{PK}_nonce_{nonce}_pick_12")
    idx = torch.arange(64, dtype=torch.int32).unsqueeze(0)
    scores = gr._batched_murmur3_32(
        idx, torch.tensor([seed], dtype=torch.int64).unsqueeze(1))
    expected = torch.topk(-scores, k=12, largest=True, sorted=False, dim=1).indices[0]
    got = gr.random_pick_indices(BH, PK, [nonce], 64, 12, DEV)[0]
    assert torch.equal(got, expected)


@pytest.mark.parametrize("nonce", [0, 7])
def test_decode_pick_differs_from_prefill(nonce):
    """Decode salts with the step, so the two schemes must not agree."""
    pre = gr.random_pick_indices(BH, PK, [nonce], 64, 12, DEV)[0]
    dec = dr.random_pick_indices_decode(BH, PK, [nonce], 64, 12, DEV, step=0)[0]
    assert not torch.equal(pre, dec)
