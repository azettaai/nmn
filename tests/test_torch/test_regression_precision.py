"""precision regressions migrated from test_issue_regressions (#160).

Original test function names are retained for issue/history traceability.
"""

from ._regression_support import (
    YatConv1D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
    YatEmbed,
    pytest,
    saturating_upcast,
    torch,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("spherical", [False, True])
def test_low_precision_conv_and_embedding_exact_match_is_finite(dtype, spherical):
    conv = YatConv1D(
        1,
        1,
        3,
        bias=False,
        use_alpha=False,
        epsilon=1e-5,
        learnable_epsilon=True,
        dtype=dtype,
    )
    with torch.no_grad():
        # The finite score is representable in fp16, but its derivative is
        # about 75,000 and exercises the saturating fp32->fp16 grad boundary.
        conv.weight.fill_(0.5)
    # Two identical patches also force the learnable-epsilon cotangent to be
    # reduced above fp16's finite range before it reaches the storage boundary.
    conv_input = torch.full((1, 1, 4), 0.5, dtype=dtype, requires_grad=True)
    conv_output = conv(conv_input)
    conv_output.sum().backward()
    assert torch.isfinite(conv_output).all() and (conv_output >= 0).all()
    assert torch.isfinite(conv_input.grad).all()
    assert torch.isfinite(conv.weight.grad).all()
    assert torch.isfinite(conv.epsilon_param.grad).all()

    embed = YatEmbed(
        1, 3, use_alpha=False, epsilon=1e-5, dtype=dtype, spherical=spherical
    )
    with torch.no_grad():
        embed.embedding.fill_(0.5)
    query = embed.embedding.detach().clone().requires_grad_()
    embed_output = embed.attend(query)
    embed_output.sum().backward()
    assert torch.isfinite(embed_output).all() and (embed_output >= 0).all()
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(embed.embedding.grad).all()


@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.float64, None])
def test_split_precision_exact_match_saturates_storage_gradients(compute_dtype):
    conv = YatConv1D(
        1,
        1,
        3,
        bias=False,
        use_alpha=False,
        epsilon=1e-5,
        learnable_epsilon=True,
        dtype=compute_dtype,
        param_dtype=torch.float16,
    )
    with torch.no_grad():
        conv.weight.fill_(0.5)
    input_dtype = torch.float64 if compute_dtype is None else compute_dtype
    x = torch.full((1, 1, 3), 0.5, dtype=input_dtype, requires_grad=True)
    output = conv(x)
    output.sum().backward()

    assert output.dtype == input_dtype and torch.isfinite(output).all()
    assert torch.isfinite(x.grad).all()
    assert conv.weight.grad.dtype == torch.float16
    assert torch.isfinite(conv.weight.grad).all()
    assert torch.isfinite(conv.epsilon_param.grad).all()


def test_saturating_upcast_preserves_nan_gradient_diagnostics():
    x = torch.ones(1, dtype=torch.float16, requires_grad=True)
    (saturating_upcast(x) * torch.tensor(float("nan"))).sum().backward()
    assert torch.isnan(x.grad).all()


@pytest.mark.parametrize("kind", ["conv", "embed"])
@pytest.mark.skipif(not hasattr(torch, "func"), reason="torch.func requires PyTorch 2")
def test_low_precision_saturating_upcast_supports_func_transforms(kind):
    if kind == "conv":
        layer = YatConv1D(1, 1, 1, bias=False, use_alpha=False, dtype=torch.float16)
        x = torch.full((2, 1, 1), 0.25, dtype=torch.float16)
        call = layer
    else:
        layer = YatEmbed(2, 1, use_alpha=False, dtype=torch.float16)
        x = torch.full((2, 1), 0.25, dtype=torch.float16)
        call = layer.attend

    primal, tangent = torch.func.jvp(call, (x,), (torch.ones_like(x),))
    batched = torch.vmap(call)(x)
    assert torch.isfinite(primal).all() and torch.isfinite(tangent).all()
    assert torch.isfinite(batched).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("conv_cls", "input_shape"),
    [
        (YatConvTranspose1D, (1, 1, 1)),
        (YatConvTranspose2D, (1, 1, 1, 1)),
        (YatConvTranspose3D, (1, 1, 1, 1, 1)),
    ],
)
def test_low_precision_transpose_conv_exact_match_is_finite(
    dtype, conv_cls, input_shape
):
    layer = conv_cls(1, 1, 1, bias=False, use_alpha=False, epsilon=1e-5, dtype=dtype)
    with torch.no_grad():
        layer.weight.fill_(0.7)
    x = torch.full(input_shape, 0.7, dtype=dtype, requires_grad=True)
    output = layer(x)
    output.sum().backward()
    assert torch.isfinite(output).all() and (output >= 0).all()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_conv_and_embedding_match_fp32_away_from_collision(dtype):
    conv32 = YatConv1D(1, 2, 3, bias=False, use_alpha=False, epsilon=1e-2)
    conv_low = YatConv1D(
        1, 2, 3, bias=False, use_alpha=False, epsilon=1e-2, dtype=dtype
    )
    conv_low.load_state_dict({k: v.to(dtype) for k, v in conv32.state_dict().items()})
    x32 = torch.tensor([[[0.2, -0.4, 0.7, 0.1, -0.3]]])
    torch.testing.assert_close(
        conv_low(x32.to(dtype)).float(), conv32(x32), rtol=6e-2, atol=2e-2
    )

    embed32 = YatEmbed(3, 4, use_alpha=False, epsilon=1e-2)
    embed_low = YatEmbed(3, 4, use_alpha=False, epsilon=1e-2, dtype=dtype)
    embed_low.load_state_dict({k: v.to(dtype) for k, v in embed32.state_dict().items()})
    query32 = torch.tensor([[0.3, -0.2, 0.6, 0.9]])
    torch.testing.assert_close(
        embed_low.attend(query32.to(dtype)).float(),
        embed32.attend(query32),
        rtol=6e-2,
        atol=2e-2,
    )
