# SPDX-License-Identifier: Apache-2.0
"""Fused Householder reflection (Triton): y = x - 2*(x·v)*v on masked rows,
y = x otherwise. Equivalent to `_reflect_torch` in native.py.

Consensus math: the bf16 rounding points match the torch reference; only the
summation order of the in-block dot differs (last bits). The kernel is static
in shape and captured into the CUDA graph, so it is compiled in `warmup()`
before capture.
"""
from __future__ import annotations

import os

import torch

try:  # Triton ships with vLLM; without it, fall back to the torch reference
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    _HAVE_TRITON = False


def fused_enabled() -> bool:
    return _HAVE_TRITON and os.environ.get("POC_FUSED_REFLECT", "1") not in ("0", "false", "no")


if _HAVE_TRITON:

    @triton.jit
    def _reflect_rows_kernel(x_ptr, v_ptr, mask_ptr, y_ptr,
                             n_rows, copies, hidden,
                             x_row_stride, v_row_stride,
                             ROUND_BF16: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.program_id(0)
        if r >= n_rows:
            return
        row = r // copies                  # source row: owns the vector and mask
        cols = tl.arange(0, BLOCK)
        cmask = cols < hidden
        x = tl.load(x_ptr + r * x_row_stride + cols, mask=cmask, other=0.0)
        m = tl.load(mask_ptr + row)
        if m != 0:
            v = tl.load(v_ptr + row * v_row_stride + cols, mask=cmask, other=0.0)
            xf = x.to(tl.float32)
            vf = v.to(tl.float32)
            prod = xf * vf
            if ROUND_BF16:
                prod = prod.to(tl.bfloat16).to(tl.float32)
            dot = tl.sum(prod, axis=0)
            if ROUND_BF16:
                dot = dot.to(tl.bfloat16).to(tl.float32)
                two_dot = (2.0 * dot).to(tl.bfloat16).to(tl.float32)
                t = (two_dot * vf).to(tl.bfloat16).to(tl.float32)
                y = (xf - t).to(tl.bfloat16)
            else:
                y = (xf - 2.0 * dot * vf).to(x.dtype)
            tl.store(y_ptr + r * x_row_stride + cols, y, mask=cmask)
        else:
            tl.store(y_ptr + r * x_row_stride + cols, x, mask=cmask)


def reflect_fused(x: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x: [n, *copies, hidden]; v: [n, hidden] (dtype x); mask: [n] bool.
    Returns a new tensor shaped like x (as torch.where in the reference)."""
    n = x.shape[0]
    hidden = x.shape[-1]
    x2 = x.contiguous().view(-1, hidden)
    copies = x2.shape[0] // n if n else 1
    v2 = v.reshape(n, hidden).contiguous()
    m8 = mask.reshape(n).to(torch.uint8)
    y = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(hidden)
    _reflect_rows_kernel[(x2.shape[0],)](
        x2, v2, m8, y, x2.shape[0], copies, hidden,
        x2.stride(0), v2.stride(0),
        ROUND_BF16=(x.dtype == torch.bfloat16), BLOCK=BLOCK,
        num_warps=8 if BLOCK >= 4096 else 4)
    return y.view(x.shape)


def warmup(hidden: int, device, dtype=torch.bfloat16) -> bool:
    """Compile the kernel before CUDA-graph capture. True if the fused path is on."""
    if not fused_enabled() or not torch.cuda.is_available():
        return False
    x = torch.zeros(2, hidden, device=device, dtype=dtype)
    v = torch.zeros(2, hidden, device=device, dtype=dtype)
    m = torch.tensor([True, False], device=device)
    reflect_fused(x, v, m)
    torch.cuda.synchronize(device)
    return True
