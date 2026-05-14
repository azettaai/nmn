"""YAT convolution layers for MLX.

Implements ``YatConv{1,2,3}D`` and ``YatConvTranspose{1,2,3}D`` on top of
``mlx.core.conv_general`` / ``mlx.core.conv_transpose{1,2,3}d``. Mirrors the
surface of ``nmn.tf.conv`` so the cross-framework parity tests can reuse
the same fixtures.

Tensor layouts (MLX native, channels-last):

* input  : ``(N, *spatial, C_in)``
* kernel : ``(C_out, *spatial, C_in / groups)``  (MLX convention)
* output : ``(N, *spatial', C_out)``

Padding accepts ``"valid"`` / ``"same"`` strings or an integer / sequence
of integers for explicit symmetric padding. Asymmetric padding for
"same" is supported via ``mx.conv_general``'s tuple form.

Limitations vs the NNX backend (to be lifted in later PRs):
* No ``CIRCULAR`` / ``REFLECT`` / ``CAUSAL`` padding modes — only
  ``valid`` / ``same`` / integer.
* No DropConnect or mask wiring.
* ``YatConvTranspose*`` is ``groups=1`` only (an MLX limitation, not a
  YAT one — ``mx.core.conv_transpose*d`` does not yet support grouped
  transposed conv).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from ._yat_core import yat_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_tuple(x: Union[int, Sequence[int]], ndim: int) -> Tuple[int, ...]:
    if isinstance(x, int):
        return (x,) * ndim
    out = tuple(int(v) for v in x)
    if len(out) != ndim:
        raise ValueError(f"expected sequence of length {ndim}, got {out}")
    return out


def _resolve_padding(
    padding: Union[str, int, Sequence[int]],
    kernel_size: Tuple[int, ...],
    dilation: Tuple[int, ...],
    input_spatial: Tuple[int, ...],
    strides: Tuple[int, ...],
) -> Tuple[List[int], List[int]]:
    """Return (low, high) padding lists per spatial axis for conv_general."""
    if isinstance(padding, str):
        mode = padding.upper()
        if mode == "VALID":
            return ([0] * len(kernel_size), [0] * len(kernel_size))
        if mode == "SAME":
            low: List[int] = []
            high: List[int] = []
            for k, s, d, in_size in zip(kernel_size, strides, dilation, input_spatial):
                out_size = (in_size + s - 1) // s
                effective_k = (k - 1) * d + 1
                total = max((out_size - 1) * s + effective_k - in_size, 0)
                lo = total // 2
                low.append(lo)
                high.append(total - lo)
            return low, high
        raise ValueError(f"unknown padding mode: {padding!r}")
    if isinstance(padding, int):
        return ([padding] * len(kernel_size), [padding] * len(kernel_size))
    pads = tuple(int(p) for p in padding)
    if len(pads) != len(kernel_size):
        raise ValueError(f"padding must have length {len(kernel_size)}")
    return (list(pads), list(pads))


# ---------------------------------------------------------------------------
# Forward-conv base
# ---------------------------------------------------------------------------


class _YatConvBase(nn.Module):
    """N-dimensional YAT convolution (forward / cross-correlation).

    Subclasses set the spatial ``_ndim`` attribute. All shared logic lives
    here so the 1D/2D/3D classes are thin validators.
    """

    _ndim: int = 0  # overridden by subclasses

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Sequence[int]],
        strides: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int]] = "valid",
        dilation_rate: Union[int, Sequence[int]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        constant_bias: Optional[float] = None,
        use_alpha: bool = True,
        constant_alpha: Optional[Union[bool, float]] = None,
        use_dropconnect: bool = False,
        drop_rate: float = 0.0,
        weight_normalized: bool = False,
        dtype: mx.Dtype = mx.float32,
        epsilon: float = 1e-5,
        learnable_epsilon: bool = False,
    ) -> None:
        super().__init__()
        if self._ndim == 0:
            raise TypeError("_YatConvBase is not meant to be instantiated directly")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if groups < 1:
            raise ValueError(f"groups must be >= 1, got {groups}")
        if not 0.0 <= drop_rate < 1.0:
            raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")

        self.filters = filters
        self.kernel_size = _as_tuple(kernel_size, self._ndim)
        self.strides = _as_tuple(strides, self._ndim)
        self.dilation_rate = _as_tuple(dilation_rate, self._ndim)
        self.groups = groups
        self.padding = padding
        self.dtype = dtype
        self.epsilon = epsilon
        self.learnable_epsilon = learnable_epsilon
        self.use_dropconnect = use_dropconnect
        self.drop_rate = drop_rate
        self.weight_normalized = weight_normalized

        # Bias configuration.
        self._constant_bias_value: Optional[float] = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            use_bias = True
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        # Alpha configuration.
        from .nmn import DEFAULT_CONSTANT_ALPHA
        self._constant_alpha_value: Optional[float] = None
        if constant_alpha is not None and constant_alpha is not False:
            if constant_alpha is True:
                self._constant_alpha_value = DEFAULT_CONSTANT_ALPHA
            else:
                self._constant_alpha_value = float(constant_alpha)
            use_alpha = True
        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha

        self.is_built = False
        self.input_channels: Optional[int] = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, input_channels: int) -> None:
        if self.is_built:
            return
        if input_channels % self.groups != 0:
            raise ValueError(
                f"Input channels ({input_channels}) must be divisible by "
                f"groups ({self.groups})"
            )
        if self.filters % self.groups != 0:
            raise ValueError(
                f"filters ({self.filters}) must be divisible by groups "
                f"({self.groups})"
            )
        self.input_channels = input_channels

        # MLX kernel layout: (C_out, *K, C_in / groups).
        channels_per_group = input_channels // self.groups
        kernel_shape = (self.filters, *self.kernel_size, channels_per_group)
        fan_in = channels_per_group
        for k in self.kernel_size:
            fan_in *= k
        std = math.sqrt(1.0 / max(fan_in, 1))
        self.kernel = (mx.random.normal(shape=kernel_shape) * std).astype(self.dtype)

        if self.use_bias and self._constant_bias_value is None:
            self.bias = mx.zeros((self.filters,), dtype=self.dtype)
        if self.use_alpha and self._constant_alpha_value is None:
            self.alpha = mx.ones((1,), dtype=self.dtype)
        if self.learnable_epsilon:
            raw_eps = math.log(math.exp(self.epsilon) - 1.0)
            self.epsilon_param = mx.array([raw_eps], dtype=self.dtype)

        self.is_built = True

    def _maybe_build(self, inputs: mx.array) -> None:
        if not self.is_built:
            self.build(inputs.shape[-1])
        elif self.input_channels != inputs.shape[-1]:
            raise ValueError(
                f"Input channels changed: expected {self.input_channels}, "
                f"got {inputs.shape[-1]}"
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(self, inputs: mx.array, *, deterministic: bool = True) -> mx.array:
        if inputs.ndim != self._ndim + 2:
            raise ValueError(
                f"expected input with {self._ndim + 2} dims "
                f"(N, *spatial, C_in), got shape {inputs.shape}"
            )
        if inputs.dtype != self.dtype:
            inputs = inputs.astype(self.dtype)
        self._maybe_build(inputs)

        spatial = inputs.shape[1:-1]
        low_pad, high_pad = _resolve_padding(
            self.padding, self.kernel_size, self.dilation_rate, spatial, self.strides
        )
        padding_arg = (low_pad, high_pad)

        kernel = self.kernel
        # Weight normalization: normalize each filter to unit norm at forward
        # time so the ||W||² term collapses to a constant 1 below.
        if self.weight_normalized:
            reduce_axes = tuple(range(1, kernel.ndim))
            norm = mx.sqrt(
                mx.sum(kernel * kernel, axis=reduce_axes, keepdims=True)
            )
            kernel = kernel / (norm + 1e-8)
        # DropConnect on the kernel.
        if self.use_dropconnect and not deterministic and self.drop_rate > 0.0:
            keep_prob = 1.0 - self.drop_rate
            dc_mask = mx.random.bernoulli(p=keep_prob, shape=kernel.shape)
            kernel = (kernel * dc_mask.astype(kernel.dtype)) / keep_prob

        # Dot product map via standard cross-correlation.
        dot_prod_map = mx.conv_general(
            inputs,
            kernel,
            stride=list(self.strides),
            padding=padding_arg,
            kernel_dilation=list(self.dilation_rate),
            groups=self.groups,
        )

        # ||x_patch||² via the ones-kernel trick. Ones kernel has one output
        # channel per group (so groups=G makes each group only see its own
        # input channels), then we broadcast back to filters.
        channels_per_group = self.input_channels // self.groups
        ones_kernel = mx.ones(
            (self.groups, *self.kernel_size, channels_per_group),
            dtype=self.dtype,
        )
        patch_sq_per_group = mx.conv_general(
            inputs * inputs,
            ones_kernel,
            stride=list(self.strides),
            padding=padding_arg,
            kernel_dilation=list(self.dilation_rate),
            groups=self.groups,
        )
        if self.groups > 1:
            # patch_sq_per_group has C_out=groups; repeat each entry
            # filters/groups times to map back to per-filter channels.
            patch_sq_map = mx.repeat(
                patch_sq_per_group, self.filters // self.groups, axis=-1
            )
        else:
            patch_sq_map = mx.repeat(patch_sq_per_group, self.filters, axis=-1)

        # ||W||² per filter — collapses to 1 in weight-normalized mode.
        if self.weight_normalized:
            kernel_sq_per_filter = mx.ones((self.filters,), dtype=kernel.dtype)
        else:
            kernel_sq_per_filter = mx.sum(
                kernel * kernel,
                axis=tuple(range(1, kernel.ndim)),
            )
        kernel_sq_shape = [1] * (dot_prod_map.ndim - 1) + [self.filters]
        kernel_sq_map = mx.reshape(kernel_sq_per_filter, kernel_sq_shape)

        distance_sq_map = mx.maximum(
            patch_sq_map + kernel_sq_map - 2.0 * dot_prod_map, 0.0
        )
        return yat_score(self, dot_prod_map, distance_sq_map)


# ---------------------------------------------------------------------------
# Forward conv subclasses
# ---------------------------------------------------------------------------


class YatConv1D(_YatConvBase):
    """1D YAT convolution. Input: ``(N, L, C_in)``."""

    _ndim = 1


class YatConv2D(_YatConvBase):
    """2D YAT convolution. Input: ``(N, H, W, C_in)``."""

    _ndim = 2


class YatConv3D(_YatConvBase):
    """3D YAT convolution. Input: ``(N, D, H, W, C_in)``."""

    _ndim = 3


# ---------------------------------------------------------------------------
# Transposed-conv base
# ---------------------------------------------------------------------------


class _YatConvTransposeBase(nn.Module):
    """N-dimensional YAT transposed convolution.

    MLX only supports ``groups=1`` for ``conv_transpose*d`` at the time of
    writing — we enforce the same restriction here.
    """

    _ndim: int = 0

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Sequence[int]],
        strides: Union[int, Sequence[int]] = 1,
        padding: Union[str, int, Sequence[int]] = "valid",
        dilation_rate: Union[int, Sequence[int]] = 1,
        output_padding: Union[int, Sequence[int]] = 0,
        use_bias: bool = True,
        constant_bias: Optional[float] = None,
        use_alpha: bool = True,
        constant_alpha: Optional[Union[bool, float]] = None,
        use_dropconnect: bool = False,
        drop_rate: float = 0.0,
        dtype: mx.Dtype = mx.float32,
        epsilon: float = 1e-5,
        learnable_epsilon: bool = False,
    ) -> None:
        super().__init__()
        if self._ndim == 0:
            raise TypeError(
                "_YatConvTransposeBase is not meant to be instantiated directly"
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if not 0.0 <= drop_rate < 1.0:
            raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")

        self.filters = filters
        self.kernel_size = _as_tuple(kernel_size, self._ndim)
        self.strides = _as_tuple(strides, self._ndim)
        self.dilation_rate = _as_tuple(dilation_rate, self._ndim)
        self.output_padding = _as_tuple(output_padding, self._ndim)
        self.padding = padding
        self.dtype = dtype
        self.epsilon = epsilon
        self.learnable_epsilon = learnable_epsilon
        self.groups = 1
        self.use_dropconnect = use_dropconnect
        self.drop_rate = drop_rate

        # Bias config.
        self._constant_bias_value: Optional[float] = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            use_bias = True
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        # Alpha config.
        from .nmn import DEFAULT_CONSTANT_ALPHA
        self._constant_alpha_value: Optional[float] = None
        if constant_alpha is not None and constant_alpha is not False:
            if constant_alpha is True:
                self._constant_alpha_value = DEFAULT_CONSTANT_ALPHA
            else:
                self._constant_alpha_value = float(constant_alpha)
            use_alpha = True
        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha

        self.is_built = False
        self.input_channels: Optional[int] = None

    # ------------------------------------------------------------------
    # Build / forward
    # ------------------------------------------------------------------

    def build(self, input_channels: int) -> None:
        if self.is_built:
            return
        self.input_channels = input_channels

        # MLX conv_transpose weight layout: (C_out, *K, C_in).
        kernel_shape = (self.filters, *self.kernel_size, input_channels)
        fan_in = input_channels
        for k in self.kernel_size:
            fan_in *= k
        std = math.sqrt(1.0 / max(fan_in, 1))
        self.kernel = (mx.random.normal(shape=kernel_shape) * std).astype(self.dtype)

        if self.use_bias and self._constant_bias_value is None:
            self.bias = mx.zeros((self.filters,), dtype=self.dtype)
        if self.use_alpha and self._constant_alpha_value is None:
            self.alpha = mx.ones((1,), dtype=self.dtype)
        if self.learnable_epsilon:
            raw_eps = math.log(math.exp(self.epsilon) - 1.0)
            self.epsilon_param = mx.array([raw_eps], dtype=self.dtype)

        self.is_built = True

    def _maybe_build(self, inputs: mx.array) -> None:
        if not self.is_built:
            self.build(inputs.shape[-1])
        elif self.input_channels != inputs.shape[-1]:
            raise ValueError(
                f"Input channels changed: expected {self.input_channels}, "
                f"got {inputs.shape[-1]}"
            )

    def _conv_transpose_fn(self):
        return {
            1: mx.conv_transpose1d,
            2: mx.conv_transpose2d,
            3: mx.conv_transpose3d,
        }[self._ndim]

    def __call__(self, inputs: mx.array, *, deterministic: bool = True) -> mx.array:
        if inputs.ndim != self._ndim + 2:
            raise ValueError(
                f"expected input with {self._ndim + 2} dims "
                f"(N, *spatial, C_in), got shape {inputs.shape}"
            )
        if inputs.dtype != self.dtype:
            inputs = inputs.astype(self.dtype)
        self._maybe_build(inputs)

        # MLX's conv_transpose*d takes integer / tuple padding. For "same"
        # we approximate symmetric padding from the kernel size — works
        # exactly for odd kernels with stride 1, slightly off otherwise.
        if isinstance(self.padding, str):
            mode = self.padding.upper()
            if mode == "VALID":
                pad_arg: Union[int, Tuple[int, ...]] = 0
            elif mode == "SAME":
                pad_arg = tuple(
                    ((k - 1) * d) // 2
                    for k, d in zip(self.kernel_size, self.dilation_rate)
                )
            else:
                raise ValueError(f"unknown padding mode: {self.padding!r}")
        else:
            pad_arg = self.padding  # type: ignore[assignment]

        conv_transpose = self._conv_transpose_fn()

        kernel = self.kernel
        if self.use_dropconnect and not deterministic and self.drop_rate > 0.0:
            keep_prob = 1.0 - self.drop_rate
            dc_mask = mx.random.bernoulli(p=keep_prob, shape=kernel.shape)
            kernel = (kernel * dc_mask.astype(kernel.dtype)) / keep_prob

        # Dot-product map.
        dot_prod_map = conv_transpose(
            inputs,
            kernel,
            stride=self.strides if self._ndim > 1 else self.strides[0],
            padding=pad_arg,
            dilation=self.dilation_rate if self._ndim > 1 else self.dilation_rate[0],
            output_padding=self.output_padding if self._ndim > 1 else self.output_padding[0],
        )

        # Per-output-position ||x_contrib||² via ones-kernel.
        ones_kernel = mx.ones(
            (self.filters, *self.kernel_size, self.input_channels),
            dtype=self.dtype,
        )
        # Sum-of-squares of the input contributions overlapping each
        # output position. Because conv_transpose with ones broadcasts the
        # input squared values into every overlap site, this gives the
        # patch-sum used for ‖x − W‖² up to the per-filter weight square.
        # We repeat over filters by reducing back to a single-channel sum.
        patch_sq_filter_map = conv_transpose(
            inputs * inputs,
            ones_kernel,
            stride=self.strides if self._ndim > 1 else self.strides[0],
            padding=pad_arg,
            dilation=self.dilation_rate if self._ndim > 1 else self.dilation_rate[0],
            output_padding=self.output_padding if self._ndim > 1 else self.output_padding[0],
        )
        # All filter channels carry the same sum (the kernel was all ones)
        # — take channel 0 and broadcast.
        patch_sq_map_one = patch_sq_filter_map[..., :1]
        patch_sq_map = mx.repeat(patch_sq_map_one, self.filters, axis=-1)

        # ||W||² per filter (uses the DropConnect-masked kernel so the
        # distance reflects the actual weights used in dot_prod_map).
        kernel_sq_per_filter = mx.sum(
            kernel * kernel,
            axis=tuple(range(1, kernel.ndim)),
        )
        kernel_sq_shape = [1] * (dot_prod_map.ndim - 1) + [self.filters]
        kernel_sq_map = mx.reshape(kernel_sq_per_filter, kernel_sq_shape)

        distance_sq_map = mx.maximum(
            patch_sq_map + kernel_sq_map - 2.0 * dot_prod_map, 0.0
        )
        return yat_score(self, dot_prod_map, distance_sq_map)


class YatConvTranspose1D(_YatConvTransposeBase):
    """1D YAT transposed convolution. Input: ``(N, L, C_in)``."""

    _ndim = 1


class YatConvTranspose2D(_YatConvTransposeBase):
    """2D YAT transposed convolution. Input: ``(N, H, W, C_in)``."""

    _ndim = 2


class YatConvTranspose3D(_YatConvTransposeBase):
    """3D YAT transposed convolution. Input: ``(N, D, H, W, C_in)``."""

    _ndim = 3


# Lower-case aliases for parity with ``nmn.tf``.
YatConv1d = YatConv1D
YatConv2d = YatConv2D
YatConv3d = YatConv3D
YatConvTranspose1d = YatConvTranspose1D
YatConvTranspose2d = YatConvTranspose2D
YatConvTranspose3d = YatConvTranspose3D


__all__ = [
    "YatConv1D", "YatConv2D", "YatConv3D",
    "YatConv1d", "YatConv2d", "YatConv3d",
    "YatConvTranspose1D", "YatConvTranspose2D", "YatConvTranspose3D",
    "YatConvTranspose1d", "YatConvTranspose2d", "YatConvTranspose3d",
]
