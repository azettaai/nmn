"""attention regressions migrated from test_issue_regressions (#160).

Original test function names are retained for issue/history traceability.
"""

from ._regression_support import (
    MultiHeadYatAttention,
    _attention_value_and_gradients,
    keras,
    np,
    pytest,
    tensor,
    to_numpy,
    yat_attention_weights,
)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_fully_masked_attention_is_zero_differentiable_and_backend_stable(dtype):
    rng = np.random.default_rng(70)
    query = tensor(rng.normal(size=(1, 2, 2, 4)).astype(np.float32), dtype=dtype)
    key = tensor(rng.normal(size=(1, 3, 2, 4)).astype(np.float32), dtype=dtype)
    value = tensor(rng.normal(size=(1, 3, 2, 5)).astype(np.float32), dtype=dtype)
    mask_np = np.array([[[[False, False, False], [True, False, True]]]])
    mask = tensor(mask_np, dtype="bool")

    output, grads = _attention_value_and_gradients(query, key, value, mask)
    weights = to_numpy(yat_attention_weights(query, key, mask=mask))

    np.testing.assert_array_equal(to_numpy(output)[:, 0], 0.0)
    np.testing.assert_array_equal(weights[..., 0, :], 0.0)
    assert all(np.all(np.isfinite(to_numpy(grad))) for grad in grads)

    # Independent NumPy oracle for the partially masked row.  Fixed operands
    # make this a parity contract shared by the JAX/Torch/TF Keras CI jobs.
    q = to_numpy(query)
    k = to_numpy(key)
    dot = np.einsum("bqhd,bkhd->bhqk", q, k)
    q_sq = np.sum(q * q, axis=-1).transpose(0, 2, 1)[..., None]
    k_sq = np.sum(k * k, axis=-1).transpose(0, 2, 1)[:, :, None, :]
    scores = dot**2 / ((np.maximum(q_sq + k_sq - 2.0 * dot, 0.0) + 1e-5) * 2.0)
    scores = np.where(mask_np, scores, -np.inf)
    row = scores[..., 1, :]
    expected = np.exp(row - np.max(row, axis=-1, keepdims=True))
    expected = np.where(mask_np[..., 1, :], expected, 0.0)
    expected /= np.sum(expected, axis=-1, keepdims=True)
    tolerance = 5e-3 if dtype == "float16" else 2e-5
    np.testing.assert_allclose(
        weights[..., 1, :], expected, rtol=tolerance, atol=tolerance
    )


def test_negative_scale_cannot_make_masked_key_win_softmax():
    query = tensor(np.ones((1, 1, 1, 4), dtype=np.float32))
    key = tensor(np.ones((1, 2, 1, 4), dtype=np.float32))
    mask = tensor([True, False], dtype="bool")
    weights = to_numpy(yat_attention_weights(query, key, mask=mask, scale=-1.0))
    np.testing.assert_array_equal(weights, [[[[1.0, 0.0]]]])


@pytest.mark.parametrize("mask_rank", [2, 4])
@pytest.mark.parametrize("cross_attention", [False, True])
def test_attention_layer_zeroes_fully_masked_rows_after_projection(
    cross_attention, mask_rank
):
    layer = MultiHeadYatAttention(embed_dim=8, num_heads=2)
    query = keras.random.normal((1, 2, 8), seed=73)
    context = keras.random.normal((1, 3, 8), seed=74)
    _ = layer(query)
    layer.out_bias.assign(np.full(layer.out_bias.shape, 3.0, dtype=np.float32))
    kv_length = 3 if cross_attention else 2
    shape = (2, kv_length) if mask_rank == 2 else (1, 1, 2, kv_length)
    mask_np = np.ones(shape, dtype=bool)
    mask_np[..., 0, :] = False
    mask = tensor(mask_np, dtype="bool")
    output = (
        layer(query, key=context, value=context, attention_mask=mask)
        if cross_attention
        else layer(query, attention_mask=mask)
    )
    np.testing.assert_array_equal(to_numpy(output)[:, 0], 0.0)
    assert np.all(np.isfinite(to_numpy(output)))
