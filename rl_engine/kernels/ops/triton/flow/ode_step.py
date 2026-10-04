# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Triton MiniMax-H3 deterministic rectified-flow Euler step (portable GPU path).

Same declared contract as ``NativeH3OdeStepOp``::

    x0     = xt + sigma * v
    r      = sigma_next / sigma             # FP32, correctly-rounded divide
    x_next = r * xt + (1 - r) * x0

Math runs in fp32 inside the kernels and is rounded back to the input dtype on
store.  The blend intentionally keeps the declared expression order; the
algebraically simplified ``xt + (sigma - sigma_next) * v`` is a different
floating-point program (see the module docstring of the PyTorch reference).

Element-wise with a per-row sigma broadcast, so Axis-A batch invariance holds
bitwise: moving a logical row inside the batch changes neither its operands nor
its bytes.

``sigma`` / ``sigma_next`` may be:

* a scalar (0-dim, ``[1]``) shared by every row, or
* per-row, with exactly one value per flattened leading row of ``xt``.

Anything else fails closed rather than broadcasting silently, because an
implicit broadcast is how video and audio step indices get shared by accident
(RFC #420 section 4).
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl
from torch import Tensor

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_BLOCK = 1024


def _validate_dtype(tensor: Tensor, name: str) -> None:
    if tensor.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {tensor.dtype}.")


def _row_geometry(tensor: Tensor) -> tuple[int, int]:
    """Flatten ``tensor`` to ``[rows, width]``; ``width`` is the last dim."""
    width = tensor.shape[-1] if tensor.dim() >= 1 else 1
    return tensor.numel() // max(width, 1), width


def _normalize_sigma(value: Any, xt: Tensor, rows: int, name: str) -> tuple[Tensor, bool]:
    """Return ``(sigma, per_row)``; ``per_row`` selects the kernel branch."""
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value, dtype=torch.float32, device=xt.device)
    if value.device != xt.device:
        raise ValueError(
            f"{name} must live on the same device as xt, got {value.device} and {xt.device}."
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {value.dtype}.")
    value = value.to(dtype=torch.float32).contiguous()
    if value.numel() == 1:
        return value.reshape(()), False
    if value.numel() == rows:
        return value.reshape(rows), True
    raise ValueError(
        f"{name} must be either a scalar or exactly one value per packed row "
        f"({rows}), got numel={value.numel()}. Refusing to broadcast implicitly: "
        "video and audio schedulers own separate sigma grids."
    )


def _validate_sigma(sigma: Tensor, sigma_next: Tensor) -> None:
    if not torch.isfinite(sigma).all() or not torch.isfinite(sigma_next).all():
        raise ValueError("sigma and sigma_next must be finite.")
    if bool((sigma <= 0).any()):
        raise ValueError(
            "sigma must be strictly positive; the terminal-zero sigma terminates "
            "the grid and is not a step input."
        )
    if bool((sigma_next < 0).any()):
        raise ValueError("sigma_next must be non-negative.")
    if bool((sigma_next > sigma).any()):
        raise ValueError(
            "sigma_next must not exceed sigma; the H3 shifted grid is monotonically " "decreasing."
        )


@triton.jit
def _ode_step_fwd_kernel(
    xt_ptr,
    v_ptr,
    s_ptr,
    sn_ptr,
    xn_ptr,
    x0_ptr,
    n_elements,
    row_width,
    ROW_SIGMA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    xt = tl.load(xt_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if ROW_SIGMA:
        row = offs // row_width
        s = tl.load(s_ptr + row, mask=mask, other=1.0).to(tl.float32)
        sn = tl.load(sn_ptr + row, mask=mask, other=0.0).to(tl.float32)
    else:
        s = tl.load(s_ptr).to(tl.float32)
        sn = tl.load(sn_ptr).to(tl.float32)
    # Correctly-rounded divide: the ratio is part of the strict contract.
    r = tl.div_rn(sn, s)
    x0 = xt + s * v
    # Declared expression order.  Do not reassociate to xt + (s - sn) * v.
    xn = r * xt + (1.0 - r) * x0
    tl.store(x0_ptr + offs, x0.to(x0_ptr.dtype.element_ty), mask=mask)
    tl.store(xn_ptr + offs, xn.to(xn_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _ode_step_bwd_kernel(
    gn_ptr,
    g0_ptr,
    s_ptr,
    sn_ptr,
    gxt_ptr,
    gv_ptr,
    n_elements,
    row_width,
    ROW_SIGMA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    gn = tl.load(gn_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g0 = tl.load(g0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if ROW_SIGMA:
        row = offs // row_width
        s = tl.load(s_ptr + row, mask=mask, other=1.0).to(tl.float32)
        sn = tl.load(sn_ptr + row, mask=mask, other=0.0).to(tl.float32)
    else:
        s = tl.load(s_ptr).to(tl.float32)
        sn = tl.load(sn_ptr).to(tl.float32)
    r = tl.div_rn(sn, s)
    one_minus_r = 1.0 - r
    # x_next = r*xt + (1-r)*x0 ; x0 = xt + s*v
    #
    # Two-operand IEEE-754 addition is commutative, so the order of the two
    # addends below cannot change a single bit (verified over 1e6 float32
    # samples).  What *does* matter is keeping the same expression program as
    # the forward: use (1 - r) * gn, never the algebraically simplified
    # (s - sn) * gn.
    g_x0 = g0 + one_minus_r * gn
    g_xt = r * gn + g_x0
    g_v = s * g_x0
    tl.store(gxt_ptr + offs, g_xt.to(gxt_ptr.dtype.element_ty), mask=mask)
    tl.store(gv_ptr + offs, g_v.to(gv_ptr.dtype.element_ty), mask=mask)


def _launch_forward(
    xt: Tensor, v: Tensor, sigma: Tensor, sigma_next: Tensor
) -> tuple[Tensor, Tensor]:
    xt_c = xt.contiguous()
    v_c = v.contiguous()
    x_next = torch.empty_like(xt_c)
    x0 = torch.empty_like(xt_c)
    n_elements = xt_c.numel()
    if n_elements == 0:
        return x_next, x0
    _, row_width = _row_geometry(xt_c)
    per_row = sigma.dim() > 0
    grid = (triton.cdiv(n_elements, _BLOCK),)
    _ode_step_fwd_kernel[grid](
        xt_c,
        v_c,
        sigma,
        sigma_next,
        x_next,
        x0,
        n_elements,
        row_width,
        ROW_SIGMA=per_row,
        BLOCK=_BLOCK,
    )
    return x_next, x0


def _launch_backward(
    grad_next: Tensor,
    grad_x0: Tensor,
    xt: Tensor,
    sigma: Tensor,
    sigma_next: Tensor,
) -> tuple[Tensor, Tensor]:
    xt_c = xt.contiguous()
    gn = grad_next.contiguous()
    g0 = grad_x0.contiguous()
    g_xt = torch.empty_like(xt_c)
    g_v = torch.empty_like(xt_c)
    n_elements = xt_c.numel()
    if n_elements == 0:
        return g_xt, g_v
    _, row_width = _row_geometry(xt_c)
    per_row = sigma.dim() > 0
    grid = (triton.cdiv(n_elements, _BLOCK),)
    _ode_step_bwd_kernel[grid](
        gn,
        g0,
        sigma,
        sigma_next,
        g_xt,
        g_v,
        n_elements,
        row_width,
        ROW_SIGMA=per_row,
        BLOCK=_BLOCK,
    )
    return g_xt, g_v


class _H3OdeStepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xt, v, sigma, sigma_next):  # type: ignore[override]
        x_next, x0 = _launch_forward(xt, v, sigma, sigma_next)
        ctx.save_for_backward(xt, sigma, sigma_next)
        return x_next, x0

    @staticmethod
    def backward(ctx, grad_next, grad_x0):  # type: ignore[override]
        xt, sigma, sigma_next = ctx.saved_tensors
        needs_xt, needs_v = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if not (needs_xt or needs_v):
            return None, None, None, None
        if grad_next is None:
            grad_next = torch.zeros_like(xt)
        if grad_x0 is None:
            grad_x0 = torch.zeros_like(xt)
        g_xt, g_v = _launch_backward(grad_next, grad_x0, xt, sigma, sigma_next)
        return (
            g_xt if needs_xt else None,
            g_v if needs_v else None,
            None,
            None,
        )


class TritonH3OdeStepOp:
    """Triton implementation of the H3 data-ward Euler step."""

    op_class = "elementwise"

    def __call__(self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> tuple[Tensor, Tensor]:
        return self.forward(xt, v, sigma, sigma_next)

    def forward(self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> tuple[Tensor, Tensor]:
        s, sn = self._prepare(xt, v, sigma, sigma_next)
        return _H3OdeStepFunction.apply(xt.contiguous(), v.contiguous(), s, sn)

    def forward_fp32(
        self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any
    ) -> tuple[Tensor, Tensor]:
        s, sn = self._prepare(xt, v, sigma, sigma_next)
        return _H3OdeStepFunction.apply(xt.float().contiguous(), v.float().contiguous(), s, sn)

    @staticmethod
    def _prepare(xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> tuple[Tensor, Tensor]:
        for name, tensor in (("xt", xt), ("v", v)):
            if tensor.device.type not in ("cuda", "hip", "xpu", "musa"):
                raise RuntimeError(
                    f"TritonH3OdeStepOp requires GPU tensors, got {name}='{tensor.device}'."
                )
            _validate_dtype(tensor, name)
        if xt.device != v.device:
            raise ValueError(f"xt and v must share a device, got {xt.device} and {v.device}.")
        if xt.shape != v.shape:
            raise ValueError(
                f"xt and v must share a shape, got {tuple(xt.shape)} and {tuple(v.shape)}."
            )
        rows, _ = _row_geometry(xt)
        s, _ = _normalize_sigma(sigma, xt, rows, "sigma")
        sn, _ = _normalize_sigma(sigma_next, xt, rows, "sigma_next")
        _validate_sigma(s, sn)
        return s, sn
