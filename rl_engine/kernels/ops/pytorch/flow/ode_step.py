# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""MiniMax-H3 deterministic rectified-flow Euler step (WS1 ground truth).

Declared strict contract (RFC #420, section 2 "Scheduler fingerprint" and
section 4 "H3-specific arithmetic rules")::

    x0     = xt + sigma * v                 # data-ward denoised estimate
    r      = sigma_next / sigma             # FP32
    x_next = r * xt + (1 - r) * x0          # FP32 Euler blend

All arithmetic is performed in FP32 and cast exactly once, at the epilogue, to
the latent storage dtype.

Do **not** rewrite the blend as ``xt + (sigma - sigma_next) * v``.  The two forms
are algebraically identical but are *not* bitwise identical in FP32, because the
reassociated expression changes the floating-point addition order and therefore
the strict trajectory.  Ablation probe H13 ("Euler arithmetic": strict arm is
"FP32 ratio/blend and declared cast", probe arm is "sample-dtype blend or
reordered expression") exists specifically to catch that substitution.

``sigma`` and ``sigma_next`` are per-row semantic inputs, not incidental
metadata: one packed batch may carry several timestep-table rows, and the video
and audio schedulers own separate grids that must never share a step index
implicitly.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_dtype(tensor: Tensor, name: str) -> None:
    if tensor.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must have dtype fp16, bf16, or fp32, got {tensor.dtype}.")


def _as_sigma(value: Any, reference: Tensor, name: str) -> Tensor:
    """Normalise a sigma argument to a contiguous FP32 tensor on ``reference``'s device."""
    if not isinstance(value, Tensor):
        value = torch.as_tensor(value, dtype=torch.float32, device=reference.device)
    if value.device != reference.device:
        raise ValueError(
            f"{name} must live on the same device as xt, got "
            f"{value.device} and {reference.device}."
        )
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point, got {value.dtype}.")
    return value.to(dtype=torch.float32).contiguous()


def _validate_sigma_shape(sigma: Tensor, sigma_next: Tensor, xt: Tensor) -> None:
    """Accept a scalar or exactly one sigma per flattened packed row.

    Refusing any other broadcast keeps the video and audio step indices from
    being merged by accident (RFC #420 section 4), and keeps the reference and
    the GPU candidate on one contract.
    """
    rows = xt.numel() // max(xt.shape[-1], 1) if xt.dim() >= 1 else 1
    for name, tensor in (("sigma", sigma), ("sigma_next", sigma_next)):
        if tensor.numel() not in (1, rows):
            raise ValueError(
                f"{name} must be either a scalar or exactly one value per packed "
                f"row ({rows}), got numel={tensor.numel()}. Refusing to broadcast "
                "implicitly: video and audio schedulers own separate sigma grids."
            )


def _validate_sigma(sigma: Tensor, sigma_next: Tensor) -> None:
    """Fail closed on geometry the pinned H3 shifted sigma grid can never produce.

    The grid is a shifted ``linspace(1, 0, steps)`` with consecutive FP32
    duplicates removed, so every real step satisfies
    ``1 >= sigma > sigma_next >= 0``.  A zero ``sigma`` is the grid terminator,
    not a step input, and must never reach this operator.
    """
    if not torch.isfinite(sigma).all():
        raise ValueError("sigma must be finite.")
    if not torch.isfinite(sigma_next).all():
        raise ValueError("sigma_next must be finite.")
    if bool((sigma <= 0).any()):
        raise ValueError(
            "sigma must be strictly positive; the terminal-zero sigma terminates "
            "the grid and is not a step input."
        )
    if bool((sigma_next < 0).any()):
        raise ValueError("sigma_next must be non-negative.")
    if bool((sigma_next > sigma).any()):
        raise ValueError(
            "sigma_next must not exceed sigma; the H3 shifted grid is " "monotonically decreasing."
        )


def _ode_step_fp32(
    xt: Tensor, v: Tensor, sigma: Tensor, sigma_next: Tensor
) -> tuple[Tensor, Tensor]:
    """Declared expression order.  Every operand here is FP32."""
    r = sigma_next / sigma
    x0 = xt + sigma * v
    x_next = r * xt + (1.0 - r) * x0
    return x_next, x0


class NativeH3OdeStepOp:
    """PyTorch reference: data-ward ``x0`` plus the FP32 Euler blend.

    Math is performed in fp32 and rounded back to the input dtype on store, the
    same dual-path contract as ``NativeSiLUOp`` / ``NativeSwiGLUOp``.
    """

    op_class = "elementwise"

    def __call__(self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> tuple[Tensor, Tensor]:
        return self.forward(xt, v, sigma, sigma_next)

    def forward(self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> tuple[Tensor, Tensor]:
        self._validate(xt, v, sigma, sigma_next)
        s = _as_sigma(sigma, xt, "sigma")
        sn = _as_sigma(sigma_next, xt, "sigma_next")
        x_next, x0 = _ode_step_fp32(xt.float(), v.float(), s, sn)
        return x_next.to(xt.dtype), x0.to(xt.dtype)

    def forward_fp32(
        self, xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any
    ) -> tuple[Tensor, Tensor]:
        """Reference path consumed by the gtest gold: FP32 in, FP32 out."""
        self._validate(xt, v, sigma, sigma_next)
        s = _as_sigma(sigma, xt, "sigma")
        sn = _as_sigma(sigma_next, xt, "sigma_next")
        return _ode_step_fp32(xt.float(), v.float(), s, sn)

    @staticmethod
    def _validate(xt: Tensor, v: Tensor, sigma: Any, sigma_next: Any) -> None:
        _validate_dtype(xt, "xt")
        _validate_dtype(v, "v")
        if xt.device != v.device:
            raise ValueError(f"xt and v must share a device, got {xt.device} and {v.device}.")
        if xt.shape != v.shape:
            raise ValueError(
                f"xt and v must share a shape, got {tuple(xt.shape)} and " f"{tuple(v.shape)}."
            )
        _validate_sigma(
            _as_sigma(sigma, xt, "sigma"),
            _as_sigma(sigma_next, xt, "sigma_next"),
        )
        _validate_sigma_shape(
            _as_sigma(sigma, xt, "sigma"),
            _as_sigma(sigma_next, xt, "sigma_next"),
            xt,
        )
