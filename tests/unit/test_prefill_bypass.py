import pytest
import torch
from torch import nn

from gonka_poc.mixed.native import _install_poc_patch, prefill_wrappers_bypassed


class _Layer(nn.Module):
    def forward(self, x):
        return x * 2


class _Wrapper(nn.Module):
    """Stands in for a PoC wrapper: calls the original forward, then marks the output."""

    def forward(self, *args, **kwargs):
        return self._inner_call(*args, **kwargs) + 1


def test_prefill_forward_skips_wrappers_and_restores_them():
    layer = _Layer()
    _install_poc_patch(layer, _Wrapper())
    x = torch.ones(3)
    assert torch.equal(layer(x), x * 2 + 1)  # wrapper active on scheduler steps

    with prefill_wrappers_bypassed():
        assert torch.equal(layer(x), x * 2)  # prefill-scheme forward: original only

    with pytest.raises(RuntimeError):
        with prefill_wrappers_bypassed():
            raise RuntimeError("forward failed")
    assert torch.equal(layer(x), x * 2 + 1)  # switch is off again, even after an error
