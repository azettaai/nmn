"""serialization device regressions migrated from test_issue_regressions (#160).

Original test function names are retained for issue/history traceability.
"""

from ._regression_support import (
    MultiHeadYatAttention,
    Path,
    YatEmbed,
    keras,
    np,
    pytest,
    to_numpy,
    tomllib,
)


@pytest.mark.parametrize("layer", [YatEmbed(8, 4), MultiHeadYatAttention(4, 2)])
def test_registered_layers_round_trip_without_custom_objects(layer):
    serialized = keras.saving.serialize_keras_object(layer)
    restored = keras.saving.deserialize_keras_object(serialized)

    assert type(restored) is type(layer)
    assert restored.get_config() == layer.get_config()
    assert serialized["registered_name"].startswith("nmn>")


def test_registered_layers_clone_and_full_model_round_trip(tmp_path):
    inputs = keras.Input((3,), dtype="int32")
    embedded = YatEmbed(
        8,
        4,
        constant_alpha=True,
        dtype="float32",
        name="yat_embed",
    )(inputs)
    outputs = MultiHeadYatAttention(
        4,
        2,
        constant_alpha=1.25,
        normalize_qk=True,
        name="yat_attention",
    )(embedded)
    model = keras.Model(inputs, outputs)
    sample = keras.ops.convert_to_tensor([[0, 1, 2]], dtype="int32")
    reference = to_numpy(model(sample))

    clone = keras.models.clone_model(model)
    clone.set_weights(model.get_weights())
    np.testing.assert_allclose(to_numpy(clone(sample)), reference, rtol=1e-6)

    path = tmp_path / "registered.keras"
    model.save(path)
    restored = keras.models.load_model(path)
    np.testing.assert_allclose(to_numpy(restored(sample)), reference, rtol=1e-6)
    assert restored.get_layer("yat_embed").constant_alpha is True
    assert restored.get_layer("yat_embed").dtype_policy.name == "float32"
    assert restored.get_layer("yat_attention").constant_alpha == 1.25


def test_keras_extra_declares_keras3_without_tensorflow():
    metadata = tomllib.loads(
        (Path(__file__).parents[2] / "pyproject.toml").read_text()
    )["project"]["optional-dependencies"]

    assert metadata["keras"] == ["keras>=3.0.0"]
    assert all(
        "tensorflow" not in requirement.lower() for requirement in metadata["keras"]
    )
    assert any("tensorflow" in requirement.lower() for requirement in metadata["tf"])
