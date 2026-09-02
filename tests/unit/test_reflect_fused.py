import os

import pytest
import torch

from gonka_poc.mixed import reflect_kernel
from gonka_poc.mixed.native import _reflect, _reflect_torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="нужен GPU")


@cuda
@pytest.mark.parametrize("n,copies,hidden", [(1, 1, 7168), (7, 4, 7168), (512, 1, 7168),
                                              (64, 1, 4096), (5, 2, 100), (3, 1, 8192)])
def test_fused_matches_reference(n, copies, hidden):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(1234 + n)
    x = torch.randn(n, copies, hidden, device=dev, generator=g).to(torch.bfloat16)
    v = torch.randn(n, hidden, device=dev, generator=g)
    v = (v / v.norm(dim=-1, keepdim=True)).to(torch.bfloat16)
    mask = torch.rand(n, device=dev, generator=g) < 0.6
    if n >= 2:
        mask[0], mask[1] = True, False
    ref = _reflect_torch(x, v.view(n, 1, hidden), mask.view(n, 1, 1))
    got = reflect_kernel.reflect_fused(x, v, mask)
    assert got.shape == x.shape and got.dtype == x.dtype
    # немаскированные строки — побитово x
    assert torch.equal(got[~mask], x[~mask])
    # маскированные — эталон с точностью bf16 (отличие только в порядке суммирования dot)
    d = (got.float() - ref.float()).abs()
    tol = 2 * torch.finfo(torch.bfloat16).eps * (ref.float().abs() + 1.0)
    assert bool((d <= tol).all()), f"max |Δ|={d.max().item():.3e}, rel={(d / (ref.float().abs() + 1e-6)).max().item():.3e}"


@cuda
def test_reflect_dispatch_and_disable(monkeypatch):
    dev = torch.device("cuda")
    x = torch.randn(4, 1, 256, device=dev).to(torch.bfloat16)
    v = torch.randn(4, 256, device=dev).to(torch.bfloat16)
    m = torch.tensor([1, 0, 1, 0], device=dev, dtype=torch.bool)
    a = _reflect(x, v.view(4, 1, 256), m.view(4, 1, 1))
    monkeypatch.setenv("POC_FUSED_REFLECT", "0")
    b = _reflect(x, v.view(4, 1, 256), m.view(4, 1, 1))
    assert torch.allclose(a.float(), b.float(), atol=2e-2, rtol=2e-2)


@cuda
def test_fused_captures_in_cuda_graph():
    dev = torch.device("cuda")
    x = torch.randn(8, 7168, device=dev).to(torch.bfloat16)
    v = torch.randn(8, 7168, device=dev).to(torch.bfloat16)
    m = torch.ones(8, device=dev, dtype=torch.bool)
    reflect_kernel.warmup(7168, dev)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        out = reflect_kernel.reflect_fused(x, v, m)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = reflect_kernel.reflect_fused(x, v, m)
    x.copy_(torch.randn(8, 7168, device=dev).to(torch.bfloat16))
    graph.replay(); torch.cuda.synchronize()
    ref = _reflect_torch(x, v, m.view(8, 1))
    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
