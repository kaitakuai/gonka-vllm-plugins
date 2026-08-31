# SPDX-License-Identifier: Apache-2.0
"""A decode round must not leave the PoC row mask set.

The in-model wrappers are gated by that mask. The prefill scheme reaches the
worker over collective_rpc and never passes through the bridge, so a mask left
set by a decode round would still be set when the prefill forward runs: on
1xH100 / Qwen3-1.7B the same request returned a different vector_b64 depending
on whether a decode round had run before it.

The decode side clears it, because the decode side is what sets it. The
prefill files stay byte-identical to 0.1.3 and know nothing about this.
"""
from gonka_poc.mixed.bridge import PoCRunnerBridge


class _Native:
    def __init__(self):
        self.mask_set_to = "untouched"

    def set_mask(self, row_mask):
        self.mask_set_to = row_mask


def _bridge_after_a_poc_step(out):
    bridge = PoCRunnerBridge.__new__(PoCRunnerBridge)
    bridge.runner = object()
    bridge.native = _Native()
    bridge._step = {"poc_metadata": [{"req_id": "r0"}]}
    import gonka_poc.mixed.runtime as rt
    rt.process_poc_outputs_from_hidden = lambda *a, **k: out
    rt.get_decode_manager = lambda runner: type(
        "M", (), {"free": staticmethod(lambda rid: None)})()
    bridge.extract(object())
    return bridge.native


def test_mask_cleared_after_a_round_with_outputs():
    native = _bridge_after_a_poc_step({"r0": object()})
    assert native.mask_set_to is None


def test_mask_cleared_even_when_the_round_emitted_nothing():
    native = _bridge_after_a_poc_step({})
    assert native.mask_set_to is None
