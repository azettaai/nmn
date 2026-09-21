"""precision regressions migrated from test_issue_regressions (#160).

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
    YatEmbed,
    input_gradient,
    keras,
    np,
    ops_cast,
    pytest,
    stable_yat_ratio,
    tensor,
    to_numpy,
)


@pytest.mark.parametrize(
    "layer_cls",
    [
        YatConv1D,
        YatConv2D,
        YatConv3D,
        YatConvTranspose1D,
        YatConvTranspose2D,
        YatConvTranspose3D,
    ],
)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_low_precision_exact_matches_are_finite_for_every_conv_family(layer_cls, dtype):
    rank = 1 if "1D" in layer_cls.__name__ else 2 if "2D" in layer_cls.__name__ else 3
    input_shape = (1,) + (1,) * rank + (1,)
    value = tensor(np.full(input_shape, 0.5), dtype)
    layer = layer_cls(
        1,
        (1,) * rank,
        use_bias=False,
        use_alpha=False,
        dtype=dtype,
    )

    # The default orthogonal initializer must build on JAX low-precision
    # policies without dispatching unsupported float16/bfloat16 LAPACK.
    layer(value)
    layer.kernel.assign(tensor(np.full(layer.kernel.shape, 0.5), dtype))
    output = layer(value)
    gradient = input_gradient(layer, value)

    assert keras.backend.standardize_dtype(output.dtype) == dtype
    assert np.all(np.isfinite(to_numpy(output)))
    assert np.all(to_numpy(output) >= 0)
    assert np.all(np.isfinite(to_numpy(gradient)))


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_low_precision_embedding_exact_match_preserves_policy_and_gradients(dtype):
    query = tensor([[0.5, -0.25, 0.75]], dtype)
    embed = YatEmbed(
        1,
        3,
        use_alpha=False,
        dtype=dtype,
        embedding_initializer="zeros",
    )
    embed(keras.ops.convert_to_tensor([0], dtype="int32"))
    embed.embedding.assign(query)
    output = embed.attend(query)

    if BACKEND == "jax":
        import jax
        import jax.numpy as jnp

        gradient = jax.grad(lambda x: jnp.sum(embed.attend(x)))(query)
    elif BACKEND == "tensorflow":
        tf = pytest.importorskip("tensorflow")
        with tf.GradientTape() as tape:
            tape.watch(query)
            loss = tf.reduce_sum(embed.attend(query))
        gradient = tape.gradient(loss, query)
    else:
        pytest.skip("gradient assertion is implemented for JAX and TensorFlow")

    assert keras.backend.standardize_dtype(output.dtype) == dtype
    assert np.all(np.isfinite(to_numpy(output)))
    assert np.all(to_numpy(output) >= 0)
    assert np.all(np.isfinite(to_numpy(gradient)))


@pytest.mark.skipif(BACKEND != "jax", reason="JAX cotangent regression")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_embed_shaped_exact_collision_reduces_epsilon_gradient_before_clipping(dtype):
    import jax
    import jax.numpy as jnp

    dot = jnp.full((2, 3), 0.5, dtype=getattr(jnp, dtype))
    distance = jnp.zeros_like(dot)
    # YatEmbed passes its configured epsilon as a scalar into this helper.
    epsilon = jnp.asarray(1e-5, dtype=getattr(jnp, dtype))

    gradient = jax.grad(lambda eps: jnp.sum(stable_yat_ratio(dot, distance, eps)))(
        epsilon
    )
    gradient32 = to_numpy(gradient, dtype=np.float32)

    assert gradient.shape == epsilon.shape
    assert np.all(np.isfinite(gradient32))
    assert np.all(gradient32 < 0)
    if dtype == "float16":
        np.testing.assert_array_equal(gradient32, np.asarray(-65504.0))
    else:
        epsilon32 = to_numpy(epsilon, dtype=np.float32)
        expected = -dot.size * 0.25 / np.square(epsilon32)
        np.testing.assert_allclose(gradient32, expected, rtol=2e-2)


@pytest.mark.skipif(BACKEND != "jax", reason="JAX cotangent regression")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_conv_exact_collision_has_finite_learnable_epsilon_gradient(dtype):
    import jax
    import jax.numpy as jnp

    value = jnp.full((1, 4, 1), 0.5, dtype=getattr(jnp, dtype))
    layer = YatConv1D(
        1,
        1,
        use_bias=False,
        use_alpha=False,
        epsilon=1e-5,
        learnable_epsilon=True,
        kernel_initializer="zeros",
        dtype=dtype,
    )
    layer(value)
    layer.kernel.assign(jnp.full(layer.kernel.shape, 0.5, dtype=getattr(jnp, dtype)))
    trainable_values = [variable.value for variable in layer.trainable_variables]
    epsilon_index = next(
        index
        for index, variable in enumerate(layer.trainable_variables)
        if variable is layer.epsilon_param
    )

    def loss(epsilon_param):
        values = list(trainable_values)
        values[epsilon_index] = epsilon_param
        output, _ = layer.stateless_call(values, layer.non_trainable_variables, value)
        return jnp.sum(output)

    gradient = jax.grad(loss)(trainable_values[epsilon_index])

    assert gradient.shape == layer.epsilon_param.shape
    assert np.all(np.isfinite(to_numpy(gradient, dtype=np.float32)))
    assert np.all(to_numpy(gradient, dtype=np.float32) < 0)


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [("float16", 3e-2, 3e-2), ("bfloat16", 8e-2, 8e-2)],
)
@pytest.mark.parametrize(
    "layer_cls",
    [
        YatConv1D,
        YatConv2D,
        YatConv3D,
        YatConvTranspose1D,
        YatConvTranspose2D,
        YatConvTranspose3D,
    ],
)
def test_low_precision_conv_families_track_fp32_off_collision(
    layer_cls, dtype, rtol, atol
):
    rank = 1 if "1D" in layer_cls.__name__ else 2 if "2D" in layer_cls.__name__ else 3
    input_shape = (1,) + (2,) * rank + (1,)
    x32 = tensor(np.full(input_shape, 0.2))
    kernel_shape = (1,) * rank + (1, 1)
    kernel32 = tensor(np.full(kernel_shape, 0.35))

    def make_conv(policy):
        layer = layer_cls(
            1,
            (1,) * rank,
            use_bias=False,
            use_alpha=False,
            dtype=policy,
            kernel_initializer="zeros",
        )
        layer(ops_cast(x32, policy))
        layer.kernel.assign(ops_cast(kernel32, policy))
        return layer

    conv32 = make_conv("float32")
    conv_low = make_conv(dtype)
    np.testing.assert_allclose(
        to_numpy(conv_low(ops_cast(x32, dtype))),
        to_numpy(conv32(x32)),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [("float16", 3e-2, 3e-2), ("bfloat16", 8e-2, 8e-2)],
)
def test_low_precision_embedding_tracks_fp32_off_collision(dtype, rtol, atol):
    embedding32 = tensor([[0.2, -0.1, 0.3], [-0.4, 0.25, 0.1]])
    query32 = tensor([[0.15, 0.35, -0.2]])

    def make_embed(policy):
        layer = YatEmbed(
            2,
            3,
            use_alpha=False,
            dtype=policy,
            embedding_initializer="zeros",
        )
        layer(keras.ops.convert_to_tensor([0], dtype="int32"))
        layer.embedding.assign(ops_cast(embedding32, policy))
        return layer

    embed32 = make_embed("float32")
    embed_low = make_embed(dtype)
    np.testing.assert_allclose(
        to_numpy(embed_low.attend(ops_cast(query32, dtype))),
        to_numpy(embed32.attend(query32)),
        rtol=rtol,
        atol=atol,
    )
