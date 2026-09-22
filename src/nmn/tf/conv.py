"""YAT convolution layers for TensorFlow."""

import math
from typing import Any, Callable, List, Optional, Tuple, Union

import tensorflow as tf

from nmn._conv_transpose import (
    canonical_same_crop_or_pad,
    canonical_transpose_config,
)
from nmn._epsilon import (
    epsilon_parameter_dtype,
    inverse_softplus,
    validate_epsilon,
    validate_epsilon_for_dtype,
)
from nmn._validation import validate_positive_int

from ._precision import reduction_safe_upcast
from ._yat_core import yat_score
from .saved_model import SingleInputSavedModelMixin


def _epsilon_variable_dtype(layer):
    dtype = tf.as_dtype(epsilon_parameter_dtype(layer.dtype))
    validate_epsilon_for_dtype(layer.epsilon, dtype)
    return dtype


def _validate_groups(filters: int, groups: int) -> None:
    """Validate the statically known grouped-convolution configuration."""
    validate_positive_int(filters, "filters")
    validate_positive_int(groups, "groups")
    if filters % groups != 0:
        raise ValueError(f"Filters ({filters}) must be divisible by groups ({groups})")


def _patch_norm_kernel(kernel_size, channels_per_group, groups, dtype):
    """Return a grouped-convolution kernel producing one norm per group."""
    return tf.ones(tuple(kernel_size) + (channels_per_group, groups), dtype=dtype)


def _grouped_convolution(inputs, kernel, groups, convolution):
    """Apply a convolution per channel group with portable gradients.

    TensorFlow's implicit grouped-convolution support differs by dimension and
    device (notably, grouped CPU gradients and grouped ``conv3d`` are not
    universally available). Explicit splitting removes those grouped-kernel
    limitations while preserving the usual contiguous group ordering. Device
    restrictions of the underlying ordinary convolution (for example some
    CPU dilated-convolution gradients) still apply.
    """
    if groups == 1:
        return convolution(inputs, kernel)
    input_groups = tf.split(inputs, groups, axis=-1)
    kernel_groups = tf.split(kernel, groups, axis=-1)
    return tf.concat(
        [
            convolution(group_inputs, group_kernel)
            for group_inputs, group_kernel in zip(input_groups, kernel_groups)
        ],
        axis=-1,
    )


def _upcast_yat_operands(inputs, kernel):
    """Accumulate low-precision convolutional YAT scores in float32."""
    if inputs.dtype in (tf.float16, tf.bfloat16):
        return reduction_safe_upcast(inputs), reduction_safe_upcast(kernel)
    return inputs, kernel


def _transpose_output_length(
    input_length, kernel_size, stride, padding, dilation, output_padding
):
    """TensorFlow-compatible implementation of the documented shape contract."""
    effective_kernel = dilation * (kernel_size - 1) + 1
    if output_padding is None:
        if padding == "SAME":
            return input_length * stride
        return input_length * stride + max(effective_kernel - stride, 0)
    if padding == "SAME":
        return input_length * stride + output_padding
    return (input_length - 1) * stride + effective_kernel + output_padding


def _adjust_transpose_same(value, adjustments):
    if adjustments is None:
        return value
    shape = tf.shape(value)
    begin = [0]
    begin.extend(max(low, 0) for low, _ in adjustments)
    begin.append(0)
    size = [shape[0]]
    size.extend(
        shape[axis] - max(low, 0) - max(high, 0)
        for axis, (low, high) in enumerate(adjustments, start=1)
    )
    size.append(shape[-1])
    value = tf.slice(value, begin, size)
    paddings = [[0, 0]]
    paddings.extend([[max(-low, 0), max(-high, 0)] for low, high in adjustments])
    paddings.append([0, 0])
    return tf.pad(value, paddings)


def _spatial_tuple(value, rank):
    return tuple(value) if isinstance(value, (list, tuple)) else (value,) * rank


class _YatConvCore(SingleInputSavedModelMixin, tf.Module):
    """Rank-generic implementation retaining public scalar/tuple configuration."""

    _rank: int = 1

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, ...]],
        strides: Union[int, Tuple[int, ...]] = 1,
        padding: str = "valid",
        dilation_rate: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        use_bias: bool = True,
        constant_bias: Optional[float] = None,
        use_alpha: bool = True,
        epsilon: float = 1e-5,
        learnable_epsilon: bool = False,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.filters = validate_positive_int(filters, "filters")
        self.kernel_size = (
            kernel_size
            if self._rank == 1 or kernel_size is None
            else _spatial_tuple(kernel_size, self._rank)
        )
        self.strides = (
            strides
            if self._rank == 1 or strides is None
            else _spatial_tuple(strides, self._rank)
        )
        self.padding = padding.upper()
        self.dilation_rate = (
            dilation_rate
            if self._rank == 1 or dilation_rate is None
            else _spatial_tuple(dilation_rate, self._rank)
        )
        _validate_groups(filters, groups)
        self.groups = groups
        self.use_alpha = use_alpha
        self.epsilon = validate_epsilon(epsilon)
        self.learnable_epsilon = learnable_epsilon
        self.dtype = dtype

        # Bias configuration: learnable, constant, or none
        self._constant_bias_value: Optional[float] = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            use_bias = True  # Bias is applied (but constant)
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        # Variables will be created in build
        self.is_built = False
        self.input_channels: Optional[int] = None
        self.kernel = None
        self.bias = None
        self.alpha = None
        self.epsilon_param = None

    @tf.Module.with_name_scope
    def build(self, input_shape: Union[List[int], tf.TensorShape]) -> None:
        """Builds the layer weights based on input shape.

        Args:
            input_shape: Shape of the input tensor [batch, length, channels].
        """
        if self.is_built:
            return

        input_channels = int(input_shape[-1])
        self.input_channels = input_channels

        if input_channels % self.groups != 0:
            raise ValueError(
                f"Input channels ({input_channels}) must be divisible by groups ({self.groups})"
            )

        # Kernel shape: [kernel_size, input_channels_per_group, filters]
        channels_per_group = input_channels // self.groups
        kernel_shape = _spatial_tuple(self.kernel_size, self._rank) + (
            channels_per_group,
            self.filters,
        )

        # Initialize kernel using orthogonal initialization
        kernel_init = tf.random.normal(kernel_shape, dtype=self.dtype)
        # Simple orthogonal-like initialization by normalizing
        kernel_init = kernel_init / tf.sqrt(
            tf.cast(
                channels_per_group
                * math.prod(_spatial_tuple(self.kernel_size, self._rank)),
                self.dtype,
            )
        )

        self.kernel = tf.Variable(
            kernel_init, trainable=True, name="kernel", dtype=self.dtype
        )

        # Initialize bias (learnable only; constant bias has no Variable)
        if self.use_bias and self._constant_bias_value is None:
            self.bias = tf.Variable(
                tf.zeros([self.filters], dtype=self.dtype), trainable=True, name="bias"
            )

        # Initialize alpha
        if self.use_alpha:
            self.alpha = tf.Variable(
                tf.ones([1], dtype=self.dtype), trainable=True, name="alpha"
            )

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            raw_eps = inverse_softplus(self.epsilon)
            self.epsilon_param = tf.Variable(
                tf.constant(raw_eps, shape=[1], dtype=_epsilon_variable_dtype(self)),
                trainable=True,
                name="epsilon_param",
            )

        self.is_built = True

    def _maybe_build(self, inputs: tf.Tensor) -> None:
        """Builds the layer if it hasn't been built yet."""
        if not self.is_built:
            self.build(inputs.shape)

    @tf.Module.with_name_scope
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward pass of the 1D YAT convolution.

        Args:
            inputs: Input tensor of shape [batch, length, channels].

        Returns:
            Output tensor after YAT convolution.
        """
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        self._maybe_build(inputs)
        assert self.input_channels is not None
        inputs, kernel = _upcast_yat_operands(inputs, self.kernel)

        # Compute dot product using standard convolution
        def convolution(x, kernel):
            if self._rank == 1:
                return tf.nn.conv1d(
                    x,
                    kernel,
                    stride=self.strides,
                    padding=self.padding,
                    dilations=self.dilation_rate,
                )
            op = tf.nn.conv2d if self._rank == 2 else tf.nn.conv3d
            return op(
                x,
                kernel,
                strides=[1] + list(_spatial_tuple(self.strides, self._rank)) + [1],
                padding=self.padding,
                dilations=[1]
                + list(_spatial_tuple(self.dilation_rate, self._rank))
                + [1],
            )

        dot_prod_map = _grouped_convolution(inputs, kernel, self.groups, convolution)

        # Compute ||input_patches||^2 using convolution with ones kernel
        inputs_squared = inputs * inputs

        # Create ones kernel for computing patch squared sums
        ones_kernel = _patch_norm_kernel(
            _spatial_tuple(self.kernel_size, self._rank),
            self.input_channels // self.groups,
            self.groups,
            inputs.dtype,
        )

        patch_sq_sum_map_raw = _grouped_convolution(
            inputs_squared,
            ones_kernel,
            self.groups,
            convolution,
        )

        # The helper convolution emits one channel per group. Repeat each
        # group's patch norm for that group's contiguous output-filter block.
        patch_sq_sum_map = tf.repeat(
            patch_sq_sum_map_raw, self.filters // self.groups, axis=-1
        )

        # Compute ||kernel||^2 per filter
        kernel_sq_sum_per_filter = tf.reduce_sum(
            kernel**2, axis=list(range(self._rank + 1))
        )  # Sum over spatial and input channel dims

        # Reshape for broadcasting: [1, 1, filters]
        kernel_sq_sum_reshaped = tf.reshape(
            kernel_sq_sum_per_filter, [1] * (self._rank + 1) + [-1]
        )

        # YAT: (dot + bias) ** 2 / (||x - W|| ** 2 + eps) * alpha
        distance_sq_map = patch_sq_sum_map + kernel_sq_sum_reshaped - 2 * dot_prod_map
        return yat_score(self, dot_prod_map, distance_sq_map)


class _YatTransposeCore(SingleInputSavedModelMixin, tf.Module):
    """Rank-generic implementation retaining public scalar/tuple configuration."""

    _rank: int = 1

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, ...]],
        strides: Union[int, Tuple[int, ...]] = 1,
        padding: str = "same",
        use_bias: bool = True,
        constant_bias: Optional[float] = None,
        use_alpha: bool = True,
        epsilon: float = 1e-5,
        learnable_epsilon: bool = False,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
        *,
        dilation_rate: Union[int, Tuple[int, ...]] = 1,
        output_padding: Optional[Union[int, Tuple[int, ...]]] = None,
    ):
        super().__init__(name=name)
        self.filters = validate_positive_int(filters, "filters")
        self.kernel_size = (
            kernel_size
            if self._rank == 1 or kernel_size is None
            else _spatial_tuple(kernel_size, self._rank)
        )
        self.strides = (
            strides
            if self._rank == 1 or strides is None
            else _spatial_tuple(strides, self._rank)
        )
        self.padding = padding.upper()
        self.dilation_rate = (
            dilation_rate
            if self._rank == 1 or dilation_rate is None
            else _spatial_tuple(dilation_rate, self._rank)
        )
        self.output_padding = (
            output_padding
            if self._rank == 1 or output_padding is None
            else _spatial_tuple(output_padding, self._rank)
        )
        if self.output_padding is not None:
            canonical_transpose_config(
                self.kernel_size,
                self.strides,
                self.padding,
                self.dilation_rate,
                self.output_padding,
            )
        self.use_alpha = use_alpha
        self.epsilon = validate_epsilon(epsilon)
        self.learnable_epsilon = learnable_epsilon
        self.dtype = dtype

        # Bias configuration: learnable, constant, or none
        self._constant_bias_value: Optional[float] = None
        if constant_bias is not None and constant_bias is not False:
            self._constant_bias_value = float(constant_bias)
            use_bias = True  # Bias is applied (but constant)
        self.use_bias = use_bias
        self.constant_bias = constant_bias

        self.is_built = False
        self.input_channels: Optional[int] = None
        self.kernel = None
        self.bias = None
        self.alpha = None
        self.epsilon_param = None

    @tf.Module.with_name_scope
    def build(self, input_shape: Union[List[int], tf.TensorShape]) -> None:
        """Builds the layer weights based on input shape.

        Args:
            input_shape: Shape of the input tensor ``[batch, length, channels]``.
        """
        if self.is_built:
            return

        input_channels = int(input_shape[-1])
        self.input_channels = input_channels

        # Kernel shape for transpose conv: [kernel_size, filters, input_channels]
        kernel_shape = _spatial_tuple(self.kernel_size, self._rank) + (
            self.filters,
            input_channels,
        )

        kernel_init = tf.random.normal(kernel_shape, dtype=self.dtype)
        kernel_init = kernel_init / tf.sqrt(
            tf.cast(
                self.filters * math.prod(_spatial_tuple(self.kernel_size, self._rank)),
                self.dtype,
            )
        )

        self.kernel = tf.Variable(
            kernel_init, trainable=True, name="kernel", dtype=self.dtype
        )

        # Learnable bias variable (skipped when constant_bias is set)
        if self.use_bias and self._constant_bias_value is None:
            self.bias = tf.Variable(
                tf.zeros([self.filters], dtype=self.dtype), trainable=True, name="bias"
            )

        if self.use_alpha:
            self.alpha = tf.Variable(
                tf.ones([1], dtype=self.dtype), trainable=True, name="alpha"
            )

        # Learnable epsilon parameter (softplus-constrained)
        if self.learnable_epsilon:
            raw_eps = inverse_softplus(self.epsilon)
            self.epsilon_param = tf.Variable(
                tf.constant(raw_eps, shape=[1], dtype=_epsilon_variable_dtype(self)),
                trainable=True,
                name="epsilon_param",
            )

        self.is_built = True

    def _maybe_build(self, inputs: tf.Tensor) -> None:
        if not self.is_built:
            self.build(inputs.shape)

    @tf.Module.with_name_scope
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply rank-generic transposed YAT with the canonical shape policy."""
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        self._maybe_build(inputs)
        inputs, kernel = _upcast_yat_operands(inputs, self.kernel)
        sizes = _spatial_tuple(self.kernel_size, self._rank)
        strides = _spatial_tuple(self.strides, self._rank)
        dilations = _spatial_tuple(self.dilation_rate, self._rank)
        output_padding = (
            None
            if self.output_padding is None
            else _spatial_tuple(self.output_padding, self._rank)
        )
        adjustments = (
            canonical_same_crop_or_pad(sizes, strides, dilations, output_padding)
            if self.padding == "SAME" and output_padding is not None
            else None
        )
        native_padding = "VALID" if adjustments else self.padding
        spatial = [
            _transpose_output_length(
                tf.shape(inputs)[i + 1],
                sizes[i],
                strides[i],
                native_padding,
                dilations[i],
                (
                    0
                    if adjustments
                    else (None if output_padding is None else output_padding[i])
                ),
            )
            for i in range(self._rank)
        ]

        def convolution(values, weights, channels):
            op = {
                1: tf.nn.conv1d_transpose,
                2: tf.nn.conv2d_transpose,
                3: tf.nn.conv3d_transpose,
            }[self._rank]
            return op(
                values,
                weights,
                output_shape=[tf.shape(inputs)[0]] + spatial + [channels],
                strides=strides[0] if self._rank == 1 else [1] + list(strides) + [1],
                padding=native_padding,
                dilations=(
                    dilations[0] if self._rank == 1 else [1] + list(dilations) + [1]
                ),
            )

        dot = _adjust_transpose_same(
            convolution(inputs, kernel, self.filters), adjustments
        )
        ones = tf.ones(sizes + (1, self.input_channels), dtype=inputs.dtype)
        patches = _adjust_transpose_same(
            convolution(inputs * inputs, ones, 1), adjustments
        )
        patches = tf.repeat(patches, self.filters, axis=-1)
        norms = tf.reduce_sum(
            kernel**2, axis=list(range(self._rank)) + [self._rank + 1]
        )
        norms = tf.reshape(norms, [1] * (self._rank + 1) + [-1])
        return yat_score(self, dot, patches + norms - 2 * dot)


class YatConv1D(_YatConvCore):
    """1D YAT convolution module using TensorFlow operations.

    This module implements 1D convolution using the YAT  algorithm,
    which computes (dot_product)^2 / (squared_euclidean_distance + epsilon).

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: Integer, specifying the length of the 1D convolution window.
        strides: Integer, specifying the stride length of the convolution. Defaults to 1.
        padding: String, either "valid" or "same" (case-insensitive). Defaults to "valid".
        dilation_rate: Integer, dilation rate to use for dilated convolution. Defaults to 1.
        groups: Integer, number of groups for grouped convolution. Defaults to 1.
        use_bias: Boolean, whether to add a bias to the output. Defaults to True.
        use_alpha: Boolean, whether to use alpha scaling. Defaults to True.
        epsilon: Float, small constant for numerical stability. Defaults to 1e-6.
        dtype: The dtype of the computation. Defaults to tf.float32.
        name: Name of the module.
    """

    _rank: int = 1

    input_channels: Optional[int]
    kernel: Optional[tf.Variable]
    bias: Optional[tf.Variable]
    alpha: Optional[tf.Variable]
    epsilon_param: Optional[tf.Variable]

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        padding: str = "valid",
        dilation_rate: int = 1,
        groups: int = 1,
        use_bias: bool = True,
        constant_bias: Optional[float] = None,
        use_alpha: bool = True,
        epsilon: float = 1e-05,
        learnable_epsilon: bool = False,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            groups=groups,
            use_bias=use_bias,
            constant_bias=constant_bias,
            use_alpha=use_alpha,
            epsilon=epsilon,
            learnable_epsilon=learnable_epsilon,
            dtype=dtype,
            name=name,
        )


class YatConv2D(_YatConvCore):
    """2D YAT convolution module using TensorFlow operations.

    This module implements 2D convolution using the YAT  algorithm,
    which computes (dot_product)^2 / (squared_euclidean_distance + epsilon).

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: Integer or tuple/list of 2 integers, specifying the height and width
            of the 2D convolution window.
        strides: Integer or tuple/list of 2 integers, specifying the strides of the convolution.
            Defaults to (1, 1).
        padding: String, either "valid" or "same" (case-insensitive). Defaults to "valid".
        dilation_rate: Integer or tuple/list of 2 integers, dilation rate for dilated convolution.
            Defaults to (1, 1).
        groups: Integer, number of groups for grouped convolution. Defaults to 1.
        use_bias: Boolean, whether to add a bias to the output. Defaults to True.
        use_alpha: Boolean, whether to use alpha scaling. Defaults to True.
        epsilon: Float, small constant for numerical stability. Defaults to 1e-6.
        dtype: The dtype of the computation. Defaults to tf.float32.
        name: Name of the module.
    """

    _rank = 2

    input_channels: Optional[int]
    kernel: Optional[tf.Variable]
    bias: Optional[tf.Variable]
    alpha: Optional[tf.Variable]
    epsilon_param: Optional[tf.Variable]

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = (1, 1),
        padding: str = "valid",
        dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
        groups: int = 1,
        use_bias: bool = True,
        constant_bias: Optional[float] = None,
        use_alpha: bool = True,
        epsilon: float = 1e-05,
        learnable_epsilon: bool = False,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            groups=groups,
            use_bias=use_bias,
            constant_bias=constant_bias,
            use_alpha=use_alpha,
            epsilon=epsilon,
            learnable_epsilon=learnable_epsilon,
            dtype=dtype,
            name=name,
        )


class YatConv3D(_YatConvCore):
    """3D YAT convolution module using TensorFlow operations.

    This module implements 3D convolution using the YAT algorithm,
    which computes (dot_product)^2 / (squared_euclidean_distance + epsilon).

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: Integer or tuple/list of 3 integers, specifying the depth, height and width
            of the 3D convolution window.
        strides: Integer or tuple/list of 3 integers, specifying the strides of the convolution.
            Defaults to (1, 1, 1).
        padding: String, either "valid" or "same" (case-insensitive). Defaults to "valid".
        dilation_rate: Integer or tuple/list of 3 integers, dilation rate for dilated convolution.
            Defaults to (1, 1, 1).
        groups: Integer, number of groups for grouped convolution. Defaults to 1.
        use_bias: Boolean, whether to add a bias to the output. Defaults to True.
        use_alpha: Boolean, whether to use alpha scaling. Defaults to True.
        epsilon: Float, small constant for numerical stability. Defaults to 1e-6.
        dtype: The dtype of the computation. Defaults to tf.float32.
        name: Name of the module.
    """

    _rank = 3

    input_channels: Optional[int]
    kernel: Optional[tf.Variable]
    bias: Optional[tf.Variable]
    alpha: Optional[tf.Variable]
    epsilon_param: Optional[tf.Variable]

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        strides: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        padding: str = "valid",
        dilation_rate: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        groups: int = 1,
        use_bias: bool = True,
        constant_bias: Optional[float] = None,
        use_alpha: bool = True,
        epsilon: float = 1e-05,
        learnable_epsilon: bool = False,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            groups=groups,
            use_bias=use_bias,
            constant_bias=constant_bias,
            use_alpha=use_alpha,
            epsilon=epsilon,
            learnable_epsilon=learnable_epsilon,
            dtype=dtype,
            name=name,
        )


class YatConvTranspose1D(_YatTransposeCore):
    """1D YAT transposed convolution (deconvolution) module using TensorFlow operations.

    This module implements 1D transposed convolution using the YAT algorithm.

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: Integer, specifying the length of the 1D convolution window.
        strides: Integer, specifying the stride length. Defaults to 1.
        padding: String, either "valid" or "same". Defaults to "same".
        dilation_rate: Kernel dilation. Defaults to 1.
        output_padding: Optional high-side extension. Passing it explicitly,
            including zero, selects the canonical NMN output-shape contract.
        use_bias: Boolean, whether to add a bias to the output. Defaults to True.
        use_alpha: Boolean, whether to use alpha scaling. Defaults to True.
        epsilon: Float, small constant for numerical stability. Defaults to 1e-6.
        dtype: The dtype of the computation. Defaults to tf.float32.
        name: Name of the module.
    """

    _rank: int = 1

    input_channels: Optional[int]
    kernel: Optional[tf.Variable]
    bias: Optional[tf.Variable]
    alpha: Optional[tf.Variable]
    epsilon_param: Optional[tf.Variable]

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        padding: str = "same",
        use_bias: bool = True,
        constant_bias: Optional[float] = None,
        use_alpha: bool = True,
        epsilon: float = 1e-05,
        learnable_epsilon: bool = False,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
        *,
        dilation_rate: int = 1,
        output_padding: Optional[int] = None,
    ) -> None:
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            constant_bias=constant_bias,
            use_alpha=use_alpha,
            epsilon=epsilon,
            learnable_epsilon=learnable_epsilon,
            dtype=dtype,
            name=name,
            dilation_rate=dilation_rate,
            output_padding=output_padding,
        )


class YatConvTranspose2D(_YatTransposeCore):
    """2D YAT transposed convolution (deconvolution) module using TensorFlow operations.

    This module implements 2D transposed convolution using the YAT algorithm.

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: Integer or tuple of 2 integers for kernel dimensions.
        strides: Integer or tuple of 2 integers. Defaults to (1, 1).
        padding: String, either "valid" or "same". Defaults to "same".
        dilation_rate: Kernel dilation. Defaults to (1, 1).
        output_padding: Optional high-side extensions. Passing it explicitly,
            including zero, selects the canonical NMN output-shape contract.
        use_bias: Boolean, whether to add a bias to the output. Defaults to True.
        use_alpha: Boolean, whether to use alpha scaling. Defaults to True.
        epsilon: Float, small constant for numerical stability. Defaults to 1e-6.
        dtype: The dtype of the computation. Defaults to tf.float32.
        name: Name of the module.
    """

    _rank = 2

    input_channels: Optional[int]
    kernel: Optional[tf.Variable]
    bias: Optional[tf.Variable]
    alpha: Optional[tf.Variable]
    epsilon_param: Optional[tf.Variable]

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = (1, 1),
        padding: str = "same",
        use_bias: bool = True,
        constant_bias: Optional[float] = None,
        use_alpha: bool = True,
        epsilon: float = 1e-05,
        learnable_epsilon: bool = False,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
        *,
        dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
        output_padding: Optional[Union[int, Tuple[int, int]]] = None,
    ) -> None:
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            constant_bias=constant_bias,
            use_alpha=use_alpha,
            epsilon=epsilon,
            learnable_epsilon=learnable_epsilon,
            dtype=dtype,
            name=name,
            dilation_rate=dilation_rate,
            output_padding=output_padding,
        )


class YatConvTranspose3D(_YatTransposeCore):
    """3D YAT transposed convolution (deconvolution) module using TensorFlow operations.

    This module implements 3D transposed convolution using the YAT algorithm.

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: Integer or tuple of 3 integers for kernel dimensions.
        strides: Integer or tuple of 3 integers. Defaults to (1, 1, 1).
        padding: String, either "valid" or "same". Defaults to "same".
        dilation_rate: Kernel dilation. Defaults to (1, 1, 1).
        output_padding: Optional high-side extensions. Passing it explicitly,
            including zero, selects the canonical NMN output-shape contract.
        use_bias: Boolean, whether to add a bias to the output. Defaults to True.
        use_alpha: Boolean, whether to use alpha scaling. Defaults to True.
        epsilon: Float, small constant for numerical stability. Defaults to 1e-6.
        dtype: The dtype of the computation. Defaults to tf.float32.
        name: Name of the module.
    """

    _rank = 3

    input_channels: Optional[int]
    kernel: Optional[tf.Variable]
    bias: Optional[tf.Variable]
    alpha: Optional[tf.Variable]
    epsilon_param: Optional[tf.Variable]

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        strides: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        padding: str = "same",
        use_bias: bool = True,
        constant_bias: Optional[float] = None,
        use_alpha: bool = True,
        epsilon: float = 1e-05,
        learnable_epsilon: bool = False,
        dtype: tf.DType = tf.float32,
        name: Optional[str] = None,
        *,
        dilation_rate: Union[int, Tuple[int, int, int]] = (1, 1, 1),
        output_padding: Optional[Union[int, Tuple[int, int, int]]] = None,
    ) -> None:
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            constant_bias=constant_bias,
            use_alpha=use_alpha,
            epsilon=epsilon,
            learnable_epsilon=learnable_epsilon,
            dtype=dtype,
            name=name,
            dilation_rate=dilation_rate,
            output_padding=output_padding,
        )


# DEPRECATED: lowercase aliases. The canonical names are the uppercase
# variants (YatConv1D, YatConv2D, ...) — they match the names exported
# from every other backend (torch / nnx / linen / keras). The lowercase
# aliases are kept for backward compatibility and will be removed in a
# future minor release.
YatConv1d = YatConv1D
YatConv2d = YatConv2D
YatConv3d = YatConv3D
YatConvTranspose1d = YatConvTranspose1D
YatConvTranspose2d = YatConvTranspose2D
YatConvTranspose3d = YatConvTranspose3D
