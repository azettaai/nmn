"""attention regressions migrated from test_issue_regressions (#160).

Original test function names are retained for issue/history traceability.
"""

from ._regression_support import (
    MultiHeadYatAttention,
    copy,
    torch,
)


def test_attention_compute_and_parameter_dtypes_round_trip():
    torch.manual_seed(0)
    layer = MultiHeadYatAttention(
        4, 2, dtype=torch.float32, param_dtype=torch.float64, dropout=0.0
    )
    x = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
    expected = layer(x, deterministic=True)
    expected.square().sum().backward()

    assert expected.dtype == torch.float32
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for parameter in layer.parameters():
        assert parameter.dtype == torch.float64
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()

    restored = MultiHeadYatAttention(
        4, 2, dtype=torch.float32, param_dtype=torch.float64, dropout=0.0
    )
    restored.load_state_dict(copy.deepcopy(layer.state_dict()))
    torch.testing.assert_close(
        restored(x.detach(), deterministic=True), expected.detach()
    )

    reference = MultiHeadYatAttention(
        4, 2, dtype=torch.float32, param_dtype=torch.float32
    )
    reference.load_state_dict(
        {key: value.float() for key, value in layer.state_dict().items()}
    )
    x_split = x.detach().clone().requires_grad_()
    x_reference = x.detach().clone().requires_grad_()
    split_output = layer(x_split, deterministic=True)
    reference_output = reference(x_reference, deterministic=True)
    torch.testing.assert_close(split_output, reference_output, rtol=1e-5, atol=1e-6)
    split_output.sum().backward()
    reference_output.sum().backward()
    torch.testing.assert_close(x_split.grad, x_reference.grad, rtol=1e-5, atol=1e-6)
