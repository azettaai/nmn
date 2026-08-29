"""Autograd-safe precision helpers shared by PyTorch YAT layers."""

from __future__ import annotations

import torch
from torch import Tensor


class _SaturatingUpcast(torch.autograd.Function):
    """Upcast low-precision values while saturating their returning gradient.

    YAT scores can have a finite fp16 forward value while their derivative is
    larger than fp16 can represent.  A normal ``Tensor.float()`` sends that
    derivative back through an fp16 cast as infinity.  This boundary keeps the
    fp32 computation unchanged and saturates only the gradient written to the
    low-precision leaf.
    """

    @staticmethod
    def forward(ctx, tensor: Tensor) -> Tensor:
        ctx.input_dtype = tensor.dtype
        return tensor.float()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor]:
        limits = torch.finfo(ctx.input_dtype)
        grad = torch.nan_to_num(
            grad_output,
            nan=0.0,
            posinf=limits.max,
            neginf=limits.min,
        ).clamp(min=limits.min, max=limits.max)
        return (grad.to(ctx.input_dtype),)


def saturating_upcast(tensor: Tensor) -> Tensor:
    """Return fp32 ``tensor`` with a finite low-precision gradient boundary."""
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return _SaturatingUpcast.apply(tensor)
    return tensor.float()
