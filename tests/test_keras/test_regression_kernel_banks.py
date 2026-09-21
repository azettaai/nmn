"""kernel banks regressions migrated from test_issue_regressions (#160).

Original test function names are retained for issue/history traceability.
"""

from ._regression_support import (
    ThreadPoolExecutor,
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
    keras,
    np,
    pytest,
    tensor,
    threading,
    to_numpy,
)


@pytest.mark.parametrize(
    ("layer_cls", "input_shape"),
    [
        (YatConv1D, (1, 5, 2)),
        (YatConv2D, (1, 5, 5, 2)),
        (YatConv3D, (1, 5, 5, 5, 2)),
        (YatConvTranspose1D, (1, 5, 2)),
        (YatConvTranspose2D, (1, 5, 5, 2)),
        (YatConvTranspose3D, (1, 5, 5, 5, 2)),
    ],
)
def test_kernel_bank_expansion_is_rejected_without_mutation(layer_cls, input_shape):
    layer_cls._KERNEL_BANKS.clear()
    rank = len(input_shape) - 2
    kwargs = dict(
        filters=2,
        kernel_size=(1,) * rank,
        tie_kernel_bank=True,
        kernel_bank_size=3,
        kernel_bank_id=f"regression-{layer_cls.__name__}",
        kernel_initializer="ones",
    )
    first = layer_cls(**kwargs)
    first(keras.ops.ones(input_shape))
    before = to_numpy(first.kernel)

    compatible = layer_cls(**{**kwargs, "filters": 1})
    compatible(keras.ops.ones(input_shape))
    assert compatible.kernel is first.kernel
    assert any(variable is first.kernel for variable in compatible.trainable_weights)

    too_large = layer_cls(**{**kwargs, "filters": 4, "kernel_bank_size": 4})
    with pytest.raises(ValueError, match="cannot be expanded in place"):
        too_large(keras.ops.ones(input_shape))

    np.testing.assert_array_equal(to_numpy(first.kernel), before)
    assert first.kernel.shape == before.shape


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
def test_kernel_bank_capacity_smaller_than_filters_is_rejected_before_state(layer_cls):
    layer_cls._KERNEL_BANKS.clear()
    rank = 1 if "1D" in layer_cls.__name__ else 2 if "2D" in layer_cls.__name__ else 3

    with pytest.raises(ValueError, match="must be greater than or equal to filters"):
        layer_cls(
            3,
            (1,) * rank,
            tie_kernel_bank=True,
            kernel_bank_size=2,
            kernel_bank_id="invalid-capacity",
        )

    assert not layer_cls._KERNEL_BANKS


def test_tied_kernel_bank_functional_save_load_preserves_sharing_and_optimizer(
    tmp_path,
):
    YatConv1D._KERNEL_BANKS.clear()
    inputs = keras.Input((5, 1))
    common = dict(
        kernel_size=1,
        tie_kernel_bank=True,
        kernel_bank_size=3,
        kernel_bank_id="functional-round-trip",
        use_bias=False,
        use_alpha=False,
        kernel_initializer="ones",
    )
    first_layer = YatConv1D(2, name="bank_first", **common)
    second_layer = YatConv1D(1, name="bank_second", **common)
    outputs = keras.layers.Concatenate()([first_layer(inputs), second_layer(inputs)])
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.SGD(0.01), loss="mse")
    sample = tensor(np.arange(5, dtype=np.float32).reshape(1, 5, 1))
    target = keras.ops.zeros((1, 5, 3))
    model.train_on_batch(sample, target)
    reference = to_numpy(model(sample))

    assert first_layer.kernel is second_layer.kernel
    assert len(first_layer.trainable_weights) == 1
    assert len(second_layer.trainable_weights) == 1
    assert len(model.trainable_variables) == 1
    iterations = int(to_numpy(model.optimizer.iterations))

    clone = keras.models.clone_model(model)
    clone.set_weights(model.get_weights())
    assert clone.get_layer("bank_first").kernel is clone.get_layer("bank_second").kernel
    np.testing.assert_allclose(to_numpy(clone(sample)), reference, rtol=1e-6)

    path = tmp_path / "tied-bank.keras"
    model.save(path)
    restored = keras.models.load_model(path)
    restored_first = restored.get_layer("bank_first")
    restored_second = restored.get_layer("bank_second")

    assert restored_first.kernel is restored_second.kernel
    assert len(restored_first.trainable_weights) == 1
    assert len(restored_second.trainable_weights) == 1
    assert len(restored.trainable_variables) == 1
    assert int(to_numpy(restored.optimizer.iterations)) == iterations
    np.testing.assert_allclose(to_numpy(restored(sample)), reference, rtol=1e-6)
    np.testing.assert_allclose(
        to_numpy(restored_first.kernel), to_numpy(first_layer.kernel), rtol=1e-6
    )


def test_tied_kernel_bank_creation_is_atomic_across_threads():
    YatConv1D._KERNEL_BANKS.clear()
    start = threading.Barrier(2)
    common = dict(
        filters=2,
        kernel_size=1,
        tie_kernel_bank=True,
        kernel_bank_size=2,
        kernel_bank_id="threaded-first-creation",
        use_bias=False,
        use_alpha=False,
        kernel_initializer="ones",
    )
    layers = [YatConv1D(**common), YatConv1D(**common)]

    def build(layer):
        start.wait(timeout=5)
        layer.build((None, 4, 1))
        return layer.kernel

    with ThreadPoolExecutor(max_workers=2) as executor:
        kernels = list(executor.map(build, layers))

    assert kernels[0] is kernels[1]
    assert layers[0]._kernel_bank_ref is layers[1]._kernel_bank_ref
    assert all(len(layer.trainable_weights) == 1 for layer in layers)


@pytest.mark.parametrize("dtype", ["float16", "mixed_float16"])
def test_tied_kernel_banks_are_separated_by_effective_dtype_policy(dtype):
    YatConv1D._KERNEL_BANKS.clear()
    common = dict(
        filters=1,
        kernel_size=1,
        tie_kernel_bank=True,
        kernel_bank_size=1,
        kernel_bank_id=f"dtype-policy-{dtype}",
        use_bias=False,
        use_alpha=False,
        kernel_initializer="ones",
    )
    float32_layer = YatConv1D(dtype="float32", **common)
    low_precision_layer = YatConv1D(dtype=dtype, **common)

    float32_output = float32_layer(tensor([[[0.25]]], "float32"))
    low_precision_output = low_precision_layer(tensor([[[0.25]]], "float16"))

    assert float32_layer.kernel is not low_precision_layer.kernel
    assert float32_layer._kernel_bank_ref is not low_precision_layer._kernel_bank_ref
    assert keras.backend.standardize_dtype(float32_output.dtype) == "float32"
    assert keras.backend.standardize_dtype(low_precision_output.dtype) == "float16"
    assert keras.backend.standardize_dtype(float32_layer.kernel.dtype) == "float32"
    expected_variable_dtype = "float32" if dtype == "mixed_float16" else "float16"
    assert (
        keras.backend.standardize_dtype(low_precision_layer.kernel.dtype)
        == expected_variable_dtype
    )
