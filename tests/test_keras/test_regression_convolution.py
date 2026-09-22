"""convolution regressions migrated from test_issue_regressions (#160).

Original test function names are retained for issue/history traceability.
"""

from ._regression_support import (
    BACKEND,
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
    input_gradient,
    keras,
    np,
    pytest,
    tensor,
    to_numpy,
)


@pytest.mark.parametrize(
    "layer_cls,input_shape,kernel,strides,expected",
    [
        (YatConvTranspose1D, (1, 3, 1), 2, 3, (1, 8, 1)),
        (YatConvTranspose2D, (1, 2, 3, 1), (2, 3), (3, 2), (1, 5, 7, 1)),
        (
            YatConvTranspose3D,
            (1, 2, 2, 2, 1),
            (2, 3, 2),
            (3, 2, 4),
            (1, 5, 5, 6, 1),
        ),
    ],
)
def test_canonical_conv_transpose_valid_shapes_and_config(
    layer_cls, input_shape, kernel, strides, expected
):
    layer = layer_cls(
        1,
        kernel,
        strides=strides,
        padding="valid",
        output_shape_mode="nmn",
        use_bias=False,
        use_alpha=False,
    )
    output = layer(keras.ops.ones(input_shape))
    assert tuple(output.shape) == expected
    assert tuple(layer.compute_output_shape(input_shape)) == expected
    restored = layer_cls.from_config(layer.get_config())
    assert restored.output_shape_mode == "nmn"


def test_canonical_same_output_padding_extends_high_side():
    layer = YatConvTranspose1D(
        1,
        2,
        strides=3,
        padding="same",
        output_padding=1,
        output_shape_mode="nmn",
        use_bias=False,
        use_alpha=False,
    )
    output = layer(keras.ops.ones((1, 3, 1)))
    assert tuple(output.shape) == (1, 10, 1)
    np.testing.assert_array_equal(to_numpy(output)[:, -1], 0.0)


def test_canonical_same_output_padding_uncrops_valid_kernel_contributions():
    inputs = keras.ops.convert_to_tensor([[[0.5], [1.0], [1.5]]])
    same = YatConvTranspose1D(
        1,
        3,
        strides=2,
        padding="same",
        output_padding=1,
        output_shape_mode="nmn",
        use_bias=False,
        use_alpha=False,
        epsilon=0.1,
    )
    valid = YatConvTranspose1D(
        1,
        3,
        strides=2,
        padding="valid",
        output_shape_mode="nmn",
        use_bias=False,
        use_alpha=False,
        epsilon=0.1,
    )
    same(inputs)
    valid(inputs)
    kernel = keras.ops.reshape(keras.ops.arange(3, dtype="float32") + 1, (3, 1, 1))
    same.kernel.assign(kernel)
    valid.kernel.assign(kernel)

    same_output = same(inputs)
    valid_output = valid(inputs)
    assert tuple(same_output.shape) == tuple(valid_output.shape) == (1, 7, 1)
    np.testing.assert_allclose(to_numpy(same_output), to_numpy(valid_output))


def test_conv_transpose_framework_mode_preserves_legacy_stride_gap():
    inputs = keras.ops.ones((1, 3, 1))
    legacy = YatConvTranspose1D(
        1, 2, strides=3, padding="valid", use_bias=False, use_alpha=False
    )
    canonical = YatConvTranspose1D(
        1,
        2,
        strides=3,
        padding="valid",
        output_shape_mode="nmn",
        use_bias=False,
        use_alpha=False,
    )
    assert tuple(legacy(inputs).shape) == (1, 9, 1)
    assert tuple(canonical(inputs).shape) == (1, 8, 1)


def test_conv_transpose_rejects_unknown_output_shape_mode():
    with pytest.raises(ValueError, match="output_shape_mode"):
        YatConvTranspose1D(1, 2, output_shape_mode="typo")


def test_canonical_conv_transpose_jax_input_and_kernel_gradients_are_finite():
    if BACKEND != "jax":
        pytest.skip("stateless parameter-gradient probe uses the JAX backend")
    import jax
    import jax.numpy as jnp

    layer = YatConvTranspose2D(
        1,
        (2, 3),
        strides=(3, 2),
        padding="valid",
        output_shape_mode="nmn",
        use_bias=False,
        use_alpha=False,
    )
    value = jnp.arange(6, dtype=jnp.float32).reshape((1, 2, 3, 1)) / 5
    layer(value)
    trainable = [variable.value for variable in layer.trainable_variables]

    def loss(x, parameters):
        output, _ = layer.stateless_call(parameters, layer.non_trainable_variables, x)
        return jnp.sum(output)

    input_grad, parameter_grads = jax.grad(loss, argnums=(0, 1))(value, trainable)
    assert jnp.isfinite(input_grad).all()
    assert all(jnp.isfinite(gradient).all() for gradient in parameter_grads)


@pytest.mark.parametrize(
    ("layer_cls", "input_shape"),
    [
        (YatConv1D, (2, 7, 4)),
        (YatConv2D, (2, 7, 6, 4)),
        (YatConv3D, (2, 6, 5, 4, 4)),
    ],
)
def test_grouped_convolutions_have_per_group_patch_norms_and_gradients(
    layer_cls, input_shape
):
    x = keras.random.normal(input_shape, seed=len(input_shape))
    layer = layer_cls(4, 3, groups=2, padding="same", use_bias=False)

    y = layer(x)
    dx = input_gradient(layer, x)

    assert y.shape[:-1] == x.shape[:-1]
    assert y.shape[-1] == 4
    assert np.all(np.isfinite(to_numpy(y)))
    assert np.all(np.isfinite(to_numpy(dx)))


def test_causal_conv1d_is_causal_and_uses_effective_dilated_padding():
    layer = YatConv1D(
        1,
        3,
        padding="causal",
        dilation_rate=2,
        use_bias=False,
        use_alpha=False,
        kernel_initializer="ones",
    )
    x_array = np.arange(1, 9, dtype=np.float32).reshape(1, 8, 1)
    changed_array = x_array.copy()
    changed_array[:, 5:, :] = 10_000.0
    x = tensor(x_array)
    changed_future = tensor(changed_array)

    y = layer(x)
    changed_y = layer(changed_future)

    assert y.shape == (1, 8, 1)
    np.testing.assert_allclose(to_numpy(y[:, :5]), to_numpy(changed_y[:, :5]))
    assert layer.compute_output_shape((None, None, 1)) == (None, None, 1)


def test_conv1d_rejects_stride_and_dilation_combination():
    with pytest.raises(ValueError, match="strides > 1"):
        YatConv1D(2, 3, strides=2, dilation_rate=2, padding="causal")


@pytest.mark.parametrize(
    ("layer_cls", "input_shape"),
    [
        (YatConv1D, (2, 9, 2)),
        (YatConv2D, (2, 9, 8, 2)),
        (YatConv3D, (2, 8, 7, 6, 2)),
        (YatConvTranspose1D, (2, 5, 2)),
        (YatConvTranspose2D, (2, 5, 4, 2)),
        (YatConvTranspose3D, (2, 4, 4, 3, 2)),
    ],
)
@pytest.mark.parametrize(
    ("padding", "stride_value", "dilation_value"),
    [("valid", 1, 2), ("same", 2, 1)],
)
def test_dilation_aware_output_shape_matches_runtime(
    layer_cls, input_shape, padding, stride_value, dilation_value
):
    rank = len(input_shape) - 2
    layer = layer_cls(
        3,
        (3,) * rank,
        padding=padding,
        strides=(stride_value,) * rank,
        dilation_rate=(dilation_value,) * rank,
        use_bias=False,
    )
    x = keras.ops.ones(input_shape, dtype="float32")

    is_transpose = layer_cls in (
        YatConvTranspose1D,
        YatConvTranspose2D,
        YatConvTranspose3D,
    )
    effective_kernel = dilation_value * 2 + 1
    if is_transpose:
        expected_spatial = tuple(
            (
                size * stride_value
                if padding == "same"
                else (size - 1) * stride_value + effective_kernel
            )
            for size in input_shape[1:-1]
        )
    else:
        expected_spatial = tuple(
            (
                (size + stride_value - 1) // stride_value
                if padding == "same"
                else (size - effective_kernel) // stride_value + 1
            )
            for size in input_shape[1:-1]
        )
    computed = tuple(layer.compute_output_shape(input_shape))
    assert computed == (input_shape[0], *expected_spatial, 3)

    # TensorFlow CPU does not implement dilated transposed convolution.  The
    # canonical shape formula above remains backend-neutral; runtime parity is
    # exercised for this case by the JAX and Torch clean-environment jobs.
    tensorflow_cpu_limitation = (
        BACKEND == "tensorflow" and is_transpose and dilation_value > 1
    )
    if not tensorflow_cpu_limitation:
        assert tuple(layer(x).shape) == computed

    unknown = (None,) + (None,) * rank + (input_shape[-1],)
    expected = (None,) + (None,) * rank + (3,)
    assert tuple(layer.compute_output_shape(unknown)) == expected
