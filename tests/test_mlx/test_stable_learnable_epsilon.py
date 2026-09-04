"""Stable learnable-epsilon handling across MLX YAT layer families."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mlx_nn = pytest.importorskip("mlx.nn")

from nmn.mlx import (  # noqa: E402
    GoatYatAttention,
    MultiHeadYatAttention,
    RotaryYatAttention,
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
    YatEmbed,
    YatNMN,
)

EPSILONS = (1e-20, 1e-5, 1000.0)
FAMILIES = (
    (YatNMN, None, (1, 2)),
    (YatConv1D, 1, (1, 1, 1)),
    (YatConv2D, (1, 1), (1, 1, 1, 1)),
    (YatConv3D, (1, 1, 1), (1, 1, 1, 1, 1)),
    (YatConvTranspose1D, 1, (1, 1, 1)),
    (YatConvTranspose2D, (1, 1), (1, 1, 1, 1)),
    (YatConvTranspose3D, (1, 1, 1), (1, 1, 1, 1, 1)),
)


def _make(layer_cls, kernel_size, epsilon, dtype=mx.float32):
    kwargs = dict(
        epsilon=epsilon,
        learnable_epsilon=True,
        use_bias=False,
        use_alpha=False,
        dtype=dtype,
    )
    if kernel_size is None:
        return layer_cls(features=1, **kwargs)
    return layer_cls(filters=1, kernel_size=kernel_size, **kwargs)


def _evaluate(layer, inputs, compiled):
    layer(inputs)
    layer.kernel = mx.full(layer.kernel.shape, 0.3, dtype=layer.dtype)

    def loss(model, values):
        output = model(values)
        return mx.sum(output.astype(mx.float32)), output

    grad_fn = mlx_nn.value_and_grad(layer, loss)
    input_grad_fn = mx.grad(lambda values: loss(layer, values)[0])
    if compiled:

        def evaluate(values):
            value, parameter_gradients = grad_fn(layer, values)
            return value, parameter_gradients, input_grad_fn(values)

        (_, output), gradients, input_gradient = mx.compile(
            evaluate, inputs=layer.state
        )(inputs)
    else:
        (_, output), gradients = grad_fn(layer, inputs)
        input_gradient = input_grad_fn(inputs)
    mx.eval(output, gradients, input_gradient)
    return output, gradients, input_gradient


@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES)
@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("compiled", [False, True])
def test_learnable_epsilon_eager_compile_and_gradients(
    layer_cls, kernel_size, input_shape, epsilon, compiled
):
    layer = _make(layer_cls, kernel_size, epsilon)
    output, gradients, input_gradient = _evaluate(
        layer, mx.full(input_shape, 0.2, dtype=mx.float32), compiled
    )

    effective = mlx_nn.softplus(layer.epsilon_param)
    mx.eval(effective)
    np.testing.assert_allclose(np.array(effective), epsilon, rtol=2e-6, atol=0.0)
    assert np.isfinite(np.array(output)).all()
    assert np.isfinite(np.array(input_gradient)).all()
    assert np.isfinite(np.array(gradients["kernel"])).all()
    epsilon_gradient = np.array(gradients["epsilon_param"])
    assert np.isfinite(epsilon_gradient).all()
    assert np.abs(epsilon_gradient).max() > 0.0


def _validation_factories(epsilon):
    return (
        lambda: YatNMN(features=1, epsilon=epsilon),
        lambda: YatConv1D(filters=1, kernel_size=1, epsilon=epsilon),
        lambda: YatConv2D(filters=1, kernel_size=1, epsilon=epsilon),
        lambda: YatConv3D(filters=1, kernel_size=1, epsilon=epsilon),
        lambda: YatConvTranspose1D(filters=1, kernel_size=1, epsilon=epsilon),
        lambda: YatConvTranspose2D(filters=1, kernel_size=1, epsilon=epsilon),
        lambda: YatConvTranspose3D(filters=1, kernel_size=1, epsilon=epsilon),
        lambda: YatEmbed(num_embeddings=2, features=2, epsilon=epsilon),
        lambda: MultiHeadYatAttention(embed_dim=4, num_heads=2, epsilon=epsilon),
        lambda: RotaryYatAttention(embed_dim=4, num_heads=2, epsilon=epsilon),
        lambda: GoatYatAttention(embed_dim=4, num_heads=2, epsilon=epsilon),
    )


@pytest.mark.parametrize(
    "epsilon", [0.0, -1.0, float("nan"), float("inf"), float("-inf")]
)
def test_every_mlx_layer_rejects_non_finite_or_non_positive_epsilon(epsilon):
    for factory in _validation_factories(epsilon):
        with pytest.raises(ValueError, match="positive and finite"):
            factory()


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("layer_cls,kernel_size,input_shape", FAMILIES)
def test_epsilon_parameter_storage_dtype_and_representability(
    layer_cls, kernel_size, input_shape, dtype, epsilon
):
    layer = _make(layer_cls, kernel_size, epsilon, dtype)
    inputs = mx.full(input_shape, 0.2, dtype=dtype)
    layer(inputs)
    expected_dtype = mx.float32 if dtype in (mx.float16, mx.bfloat16) else dtype
    assert layer.epsilon_param.dtype == expected_dtype
    effective = mlx_nn.softplus(layer.epsilon_param)
    mx.eval(effective)
    np.testing.assert_allclose(np.array(effective), epsilon, rtol=2e-6, atol=0.0)
    if dtype in (mx.float16, mx.bfloat16):
        output, gradients, input_gradient = _evaluate(layer, inputs, False)
        assert output.dtype == dtype
        assert np.isfinite(np.array(output.astype(mx.float32))).all()
        assert np.isfinite(np.array(input_gradient.astype(mx.float32))).all()
        epsilon_gradient = np.array(gradients["epsilon_param"].astype(mx.float32))
        assert np.isfinite(epsilon_gradient).all()
        assert np.abs(epsilon_gradient).max() > 0.0


@pytest.mark.parametrize("epsilon", [5e-324, 1e-46, 1e39])
def test_float32_rejects_unrepresentable_epsilon(epsilon):
    layer = _make(YatNMN, None, epsilon, mx.float32)
    with pytest.raises(ValueError, match="not representable"):
        layer(mx.ones((1, 2), dtype=mx.float32))


@pytest.mark.parametrize("epsilon", EPSILONS)
def test_dense_weight_serialization_roundtrip(tmp_path, epsilon):
    layer = _make(YatNMN, None, epsilon, mx.float16)
    inputs = mx.ones((1, 2), dtype=mx.float16)
    layer(inputs)
    layer.epsilon_param = layer.epsilon_param + mx.array([0.5], dtype=mx.float32)
    mx.eval(layer.epsilon_param)
    path = tmp_path / "epsilon.npz"
    layer.save_weights(str(path))

    restored = _make(YatNMN, None, epsilon, mx.float16)
    restored(inputs)
    mx.eval(restored.epsilon_param)
    assert not np.array_equal(
        np.array(restored.epsilon_param), np.array(layer.epsilon_param)
    )
    restored.load_weights(str(path))
    source_effective = mlx_nn.softplus(layer.epsilon_param)
    restored_effective = mlx_nn.softplus(restored.epsilon_param)
    mx.eval(restored.parameters(), source_effective, restored_effective)
    assert restored.epsilon_param.dtype == mx.float32
    np.testing.assert_array_equal(
        np.array(restored.epsilon_param), np.array(layer.epsilon_param)
    )
    np.testing.assert_array_equal(
        np.array(restored_effective), np.array(source_effective)
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("epsilon", [1e-20, 1e5])
def test_dtype_extremes_eager_compile_and_metal(dtype, epsilon, mlx_gpu):
    del mlx_gpu
    layer = _make(YatNMN, None, epsilon, dtype)
    inputs = mx.full((1, 2), 0.2, dtype=dtype)
    eager_output, eager_gradients, eager_input_gradient = _evaluate(
        layer, inputs, False
    )
    compiled_output, compiled_gradients, compiled_input_gradient = _evaluate(
        layer, inputs, True
    )

    assert eager_output.dtype == compiled_output.dtype == dtype
    for value in (
        eager_output,
        compiled_output,
        eager_input_gradient,
        compiled_input_gradient,
        eager_gradients["epsilon_param"],
        compiled_gradients["epsilon_param"],
    ):
        assert np.isfinite(np.array(value.astype(mx.float32))).all()
    np.testing.assert_allclose(
        np.array(compiled_output.astype(mx.float32)),
        np.array(eager_output.astype(mx.float32)),
        rtol=2e-2,
        atol=2e-3,
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize("epsilon", EPSILONS)
@pytest.mark.parametrize("compiled", [False, True])
def test_goat_uses_stable_epsilon_parameter(dtype, epsilon, compiled):
    layer = GoatYatAttention(
        embed_dim=4,
        num_heads=2,
        epsilon=epsilon,
        dtype=dtype,
    )
    expected_dtype = mx.float32 if dtype in (mx.float16, mx.bfloat16) else dtype
    assert layer.eps_raw.dtype == expected_dtype
    effective = mlx_nn.softplus(layer.eps_raw)
    inputs = mx.array(
        [[[0.1, 0.2, -0.3, 0.4], [0.5, -0.2, 0.1, 0.3], [-0.4, 0.2, 0.6, 0.1]]],
        dtype=dtype,
    )

    def loss(model, values):
        output = model(values, self_mask=False)
        return mx.sum(output.astype(mx.float32)), output

    grad_fn = mlx_nn.value_and_grad(layer, loss)
    if compiled:
        (_, output), gradients = mx.compile(
            lambda values: grad_fn(layer, values), inputs=layer.state
        )(inputs)
    else:
        (_, output), gradients = grad_fn(layer, inputs)
    mx.eval(effective, output, gradients)
    np.testing.assert_allclose(np.array(effective), epsilon, rtol=2e-6, atol=0.0)
    assert output.dtype == dtype
    assert np.isfinite(np.array(output.astype(mx.float32))).all()
    epsilon_gradient = np.array(gradients["eps_raw"].astype(mx.float32))
    assert np.isfinite(epsilon_gradient).all()
    if epsilon == 1e-5:
        assert np.abs(epsilon_gradient).max() > 0.0


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("epsilon", [1e-20, 1e5])
def test_goat_low_precision_compile_and_gradient_on_metal(dtype, epsilon, mlx_gpu):
    del mlx_gpu
    layer = GoatYatAttention(
        embed_dim=4,
        num_heads=2,
        epsilon=epsilon,
        dtype=dtype,
    )
    inputs = mx.array(
        [[[0.1, 0.2, -0.3, 0.4], [0.5, -0.2, 0.1, 0.3], [-0.4, 0.2, 0.6, 0.1]]],
        dtype=dtype,
    )
    grad_fn = mlx_nn.value_and_grad(
        layer,
        lambda model, values: mx.sum(model(values, self_mask=False).astype(mx.float32)),
    )
    _, gradients = mx.compile(
        lambda values: grad_fn(layer, values), inputs=layer.state
    )(inputs)
    mx.eval(gradients)
    assert layer.eps_raw.dtype == mx.float32
    assert np.isfinite(np.array(gradients["eps_raw"])).all()
