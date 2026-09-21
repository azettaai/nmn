"""kernel banks regressions migrated from test_issue_regressions (#160).

Original test function names are retained for issue/history traceability.
"""

from ._regression_support import (
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatNMN,
    pytest,
    threading,
    torch,
)


def test_tied_dense_construction_preserves_live_peer():
    YatNMN._KERNEL_BANKS.clear()
    first = YatNMN(
        4,
        2,
        tie_kernel_bank=True,
        kernel_bank_size=3,
        kernel_bank_id="issue-71",
    )
    x = torch.randn(3, 4)
    weight_before = first.weight.detach().clone()
    output_before = first(x).detach().clone()

    second = YatNMN(4, 3, tie_kernel_bank=True, kernel_bank_id="issue-71")

    torch.testing.assert_close(first.weight, weight_before)
    torch.testing.assert_close(first(x), output_before)
    assert second.weight is first.weight
    assert first.weight.requires_grad


def test_tied_dense_auto_expands_before_first_use_and_preserves_existing_slice():
    YatNMN._KERNEL_BANKS.clear()
    first = YatNMN(
        4,
        2,
        tie_kernel_bank=True,
        alpha=False,
        bias=False,
        kernel_bank_id="issue-71-pre-use-expand",
    )
    existing_slice = first.weight.detach().clone()

    second = YatNMN(
        4,
        3,
        tie_kernel_bank=True,
        alpha=False,
        bias=False,
        kernel_bank_id="issue-71-pre-use-expand",
    )

    assert first.weight is second.weight
    assert first.weight.shape == (3, 4)
    torch.testing.assert_close(first.weight[:2], existing_slice)


def test_tied_dense_rejects_expansion_without_mutating_stale_gradient():
    YatNMN._KERNEL_BANKS.clear()
    first = YatNMN(
        4,
        2,
        tie_kernel_bank=True,
        alpha=False,
        bias=False,
        kernel_bank_id="issue-71-stale-grad",
    )
    first(torch.randn(3, 4)).sum().backward()
    parameter = first.weight
    value_before = parameter.detach().clone()
    gradient_before = parameter.grad.detach().clone()

    with pytest.raises(ValueError, match="capacity is frozen"):
        YatNMN(
            4,
            3,
            tie_kernel_bank=True,
            alpha=False,
            bias=False,
            kernel_bank_id="issue-71-stale-grad",
        )

    assert first.weight is parameter
    torch.testing.assert_close(first.weight, value_before)
    torch.testing.assert_close(first.weight.grad, gradient_before)


def test_tied_dense_rejects_expansion_without_mutating_adam_state():
    YatNMN._KERNEL_BANKS.clear()
    first = YatNMN(
        4,
        2,
        tie_kernel_bank=True,
        alpha=False,
        bias=False,
        kernel_bank_id="issue-71-adam",
    )
    optimizer = torch.optim.Adam(first.parameters(), lr=1e-3)
    first(torch.randn(3, 4)).sum().backward()
    optimizer.step()
    parameter = first.weight
    value_before = parameter.detach().clone()
    state_before = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in optimizer.state[parameter].items()
    }

    with pytest.raises(ValueError, match="capacity is frozen"):
        YatNMN(
            4,
            3,
            tie_kernel_bank=True,
            alpha=False,
            bias=False,
            kernel_bank_id="issue-71-adam",
        )

    assert first.weight is parameter
    torch.testing.assert_close(first.weight, value_before)
    for key, expected in state_before.items():
        actual = optimizer.state[parameter][key]
        if torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected)
        else:
            assert actual == expected


def test_tied_dense_rejects_incompatible_lazy_consumer():
    YatNMN._KERNEL_BANKS.clear()
    first = YatNMN(4, 2, tie_kernel_bank=True, kernel_bank_id="issue-71-lazy")
    with pytest.raises(ValueError, match="same lazy"):
        YatNMN(4, 2, tie_kernel_bank=True, lazy=True, kernel_bank_id="issue-71-lazy")
    assert first.weight.requires_grad


@pytest.mark.parametrize(
    ("conv_cls", "input_shape"),
    [
        (YatConv1D, (2, 2, 7)),
        (YatConv2D, (2, 2, 5, 5)),
        (YatConv3D, (2, 2, 4, 4, 4)),
    ],
)
def test_tied_conv_bank_accumulates_gradients_and_uses_actual_bias_width(
    conv_cls, input_shape
):
    conv_cls._KERNEL_BANKS.clear()
    bank_id = f"issue-72-{conv_cls.__name__}"
    narrow = conv_cls(
        2,
        2,
        1,
        tie_kernel_bank=True,
        kernel_bank_size=4,
        kernel_bank_id=bank_id,
    )
    wide = conv_cls(
        2,
        4,
        1,
        tie_kernel_bank=True,
        kernel_bank_size=4,
        kernel_bank_id=bank_id,
    )
    assert narrow.weight is wide.weight
    assert narrow.out_channels == 2
    assert wide.out_channels == 4
    assert narrow.bias.shape == (2,)
    assert wide.bias.shape == (4,)

    reference = conv_cls(2, 2, 1)
    with torch.no_grad():
        reference.weight.copy_(narrow.weight[:2])
        reference.bias.copy_(narrow.bias)
        reference.alpha.copy_(narrow.alpha)
    x_tied = torch.randn(*input_shape, requires_grad=True)
    x_reference = x_tied.detach().clone().requires_grad_()
    tied_output = narrow(x_tied)
    reference_output = reference(x_reference)
    torch.testing.assert_close(tied_output, reference_output)
    tied_output.sum().backward()
    reference_output.sum().backward()
    torch.testing.assert_close(x_tied.grad, x_reference.grad)
    torch.testing.assert_close(narrow.weight.grad[:2], reference.weight.grad)

    narrow.zero_grad(set_to_none=True)
    wide.zero_grad(set_to_none=True)
    optimizer = torch.optim.SGD(narrow.parameters(), lr=1e-3)
    before = narrow.weight.detach().clone()
    x = torch.randn(*input_shape)
    (narrow(x).sum() + wide(x).sum()).backward()
    assert narrow.weight.grad is not None
    assert torch.isfinite(narrow.weight.grad).all()
    assert torch.count_nonzero(narrow.weight.grad[2:]) > 0
    optimizer.step()
    assert not torch.equal(before, narrow.weight)


@pytest.mark.parametrize("conv_cls", [YatConv1D, YatConv2D, YatConv3D])
def test_tied_conv_auto_expands_before_first_use_and_preserves_slice(conv_cls):
    conv_cls._KERNEL_BANKS.clear()
    bank_id = f"issue-72-pre-use-expand-{conv_cls.__name__}"
    first = conv_cls(
        2,
        2,
        1,
        tie_kernel_bank=True,
        bias=False,
        use_alpha=False,
        kernel_bank_id=bank_id,
    )
    existing_slice = first.weight.detach().clone()

    second = conv_cls(
        2,
        4,
        1,
        tie_kernel_bank=True,
        bias=False,
        use_alpha=False,
        kernel_bank_id=bank_id,
    )

    assert first.weight is second.weight
    assert first.weight.shape[0] == 4
    torch.testing.assert_close(first.weight[:2], existing_slice)
    assert first.out_channels == 2
    assert second.out_channels == 4


@pytest.mark.parametrize(
    ("conv_cls", "input_shape"),
    [
        (YatConv1D, (1, 2, 3)),
        (YatConv2D, (1, 2, 2, 2)),
        (YatConv3D, (1, 2, 2, 2, 2)),
    ],
)
def test_tied_conv_rejects_expansion_without_mutating_adam_state(conv_cls, input_shape):
    conv_cls._KERNEL_BANKS.clear()
    bank_id = f"issue-72-immutable-{conv_cls.__name__}"
    first = conv_cls(
        2,
        2,
        1,
        tie_kernel_bank=True,
        bias=False,
        use_alpha=False,
        kernel_bank_id=bank_id,
    )
    optimizer = torch.optim.Adam(first.parameters(), lr=1e-3)
    first(torch.randn(*input_shape)).sum().backward()
    optimizer.step()
    parameter = first.weight
    value_before = parameter.detach().clone()
    state_before = {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in optimizer.state[parameter].items()
    }

    with pytest.raises(ValueError, match="capacity is frozen"):
        conv_cls(
            2,
            4,
            1,
            tie_kernel_bank=True,
            bias=False,
            use_alpha=False,
            kernel_bank_id=bank_id,
        )

    assert first.weight is parameter
    assert first.out_channels == 2
    torch.testing.assert_close(first.weight, value_before)
    for key, expected in state_before.items():
        torch.testing.assert_close(optimizer.state[parameter][key], expected)


@pytest.mark.parametrize("conv_cls", [YatConv1D, YatConv2D, YatConv3D])
def test_tied_conv_bank_is_device_scoped_and_preserves_public_width(conv_cls):
    conv_cls._KERNEL_BANKS.clear()
    bank_id = f"issue-72-device-{conv_cls.__name__}"
    cpu_layer = conv_cls(
        2,
        2,
        1,
        tie_kernel_bank=True,
        kernel_bank_size=4,
        kernel_bank_id=bank_id,
        device="cpu",
    )
    meta_layer = conv_cls(
        2,
        2,
        1,
        tie_kernel_bank=True,
        kernel_bank_size=4,
        kernel_bank_id=bank_id,
        device="meta",
    )

    assert cpu_layer.weight.device.type == "cpu"
    assert meta_layer.weight.device.type == "meta"
    assert cpu_layer.weight is not meta_layer.weight
    assert cpu_layer.out_channels == meta_layer.out_channels == 2


@pytest.mark.parametrize("conv_cls", [YatConv1D, YatConv3D])
def test_tied_conv_concurrent_construction_is_atomic(conv_cls):
    conv_cls._KERNEL_BANKS.clear()
    bank_id = f"issue-72-construction-race-{conv_cls.__name__}"
    barrier = threading.Barrier(3)
    layers = []
    errors = []

    def construct(width):
        barrier.wait()
        try:
            layers.append(
                conv_cls(
                    2,
                    width,
                    1,
                    tie_kernel_bank=True,
                    bias=False,
                    use_alpha=False,
                    kernel_bank_id=bank_id,
                )
            )
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = [
        threading.Thread(target=construct, args=(2,)),
        threading.Thread(target=construct, args=(4,)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(layers) == 2
    assert layers[0].weight is layers[1].weight
    assert layers[0].weight.shape[0] == 4


@pytest.mark.parametrize(
    ("conv_cls", "input_shape"),
    [(YatConv1D, (1, 2, 3)), (YatConv3D, (1, 2, 2, 2, 2))],
)
def test_tied_conv_first_use_and_expansion_race_is_serialized(conv_cls, input_shape):
    conv_cls._KERNEL_BANKS.clear()
    bank_id = f"issue-72-use-race-{conv_cls.__name__}"
    first = conv_cls(
        2,
        2,
        1,
        tie_kernel_bank=True,
        bias=False,
        use_alpha=False,
        kernel_bank_id=bank_id,
    )
    parameter = first.weight
    barrier = threading.Barrier(3)
    outputs = []
    expanded = []
    errors = []

    def execute():
        barrier.wait()
        outputs.append(first(torch.randn(*input_shape)))

    def expand():
        barrier.wait()
        try:
            expanded.append(
                conv_cls(
                    2,
                    4,
                    1,
                    tie_kernel_bank=True,
                    bias=False,
                    use_alpha=False,
                    kernel_bank_id=bank_id,
                )
            )
        except ValueError as error:
            errors.append(error)

    threads = [threading.Thread(target=execute), threading.Thread(target=expand)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert len(outputs) == 1 and torch.isfinite(outputs[0]).all()
    assert first.weight is parameter
    assert len(expanded) + len(errors) == 1
    if expanded:
        assert expanded[0].weight is parameter
        assert parameter.shape[0] == 4
    else:
        assert "capacity is frozen" in str(errors[0])
        assert parameter.shape[0] == 2


@pytest.mark.parametrize(
    "factory",
    [
        lambda tied: YatNMN(4, 2, tie_kernel_bank=tied),
        lambda tied: YatConv1D(2, 2, 1, tie_kernel_bank=tied),
        lambda tied: YatConv2D(2, 2, 1, tie_kernel_bank=tied),
        lambda tied: YatConv3D(2, 2, 1, tie_kernel_bank=tied),
    ],
)
def test_tied_bank_rejects_apply_migration_but_untied_module_migrates(factory):
    tied = factory(True)
    parameter = tied.weight
    with pytest.raises(RuntimeError, match="migration is unsupported"):
        tied.double()
    with pytest.raises(RuntimeError, match="migration is unsupported"):
        tied.to("meta")
    assert tied.weight is parameter
    assert tied.weight.device.type == "cpu"
    assert tied.weight.dtype == torch.float32

    untied = factory(False).double()
    assert untied.weight.dtype == torch.float64


def test_tied_conv_attachment_rejects_stale_registry_dtype():
    YatConv1D._KERNEL_BANKS.clear()
    first = YatConv1D(
        2,
        2,
        1,
        tie_kernel_bank=True,
        kernel_bank_id="issue-72-stale-registry",
    )
    first.weight.data = first.weight.data.double()

    with pytest.raises(RuntimeError, match="registry is stale"):
        YatConv1D(
            2,
            2,
            1,
            tie_kernel_bank=True,
            kernel_bank_id="issue-72-stale-registry",
        )


def test_tied_yat_nmn_banks_are_device_separated_and_constructor_route_runs():
    YatNMN._KERNEL_BANKS.clear()
    bank_id = "issue-71-device-constructor"
    cpu_layer = YatNMN(
        4,
        2,
        tie_kernel_bank=True,
        learnable_epsilon=True,
        kernel_bank_id=bank_id,
        device="cpu",
        dtype=torch.float32,
        param_dtype=torch.float64,
    )
    meta_layer = YatNMN(
        4,
        2,
        tie_kernel_bank=True,
        learnable_epsilon=True,
        kernel_bank_id=bank_id,
        device="meta",
        dtype=torch.float32,
        param_dtype=torch.float64,
    )

    assert cpu_layer.weight is not meta_layer.weight
    assert all(value.device.type == "cpu" for value in cpu_layer.state_dict().values())
    assert all(
        value.device.type == "meta" for value in meta_layer.state_dict().values()
    )
    output = cpu_layer(torch.randn(3, 4, dtype=torch.float32))
    assert output.device.type == "cpu"
    assert output.dtype == torch.float32


def test_untied_yat_nmn_supports_constructor_device_and_later_migration():
    layer = YatNMN(4, 2, learnable_epsilon=True, device="cpu")
    migrated = layer.to(dtype=torch.float64)
    assert migrated is layer
    state = layer.state_dict()
    assert state["epsilon_param"].dtype == torch.float32
    assert all(
        value.dtype == torch.float64
        for name, value in state.items()
        if name != "epsilon_param"
    )
