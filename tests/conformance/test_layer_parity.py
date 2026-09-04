"""Canonical lookup and 1D convolution parity across executable adapters."""

from __future__ import annotations

import pytest

from nmn.conformance import load_contract
from tests.conformance.harness import compare, load_adapter
from tests.conformance.oracle import (
    canonical_convolution_case,
    canonical_embedding_attend_case,
    canonical_embedding_case,
    canonical_transpose_convolution_case,
    yat_conv1d,
    yat_conv_transpose1d,
    yat_embedding,
    yat_embedding_attend,
)

# These are the locally installed frameworks whose native layers are exercised
# by this compact representative matrix.  Optional backends retain their own
# backend suites until an adapter is added and executed in their native CI.
ADAPTERS = {
    "torch": "tests.conformance.adapters.torch:TorchAdapter",
    "nnx": "tests.conformance.adapters.nnx:NnxAdapter",
    "linen": "tests.conformance.adapters.linen:LinenAdapter",
}
TOLERANCE = load_contract()["tolerances"]["float32"]


@pytest.mark.parametrize("backend,reference", ADAPTERS.items())
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_embedding_lookup_and_embedding_gradient_match_float64_oracle(
    backend, reference, compiled
):
    adapter = load_adapter(reference)
    if not adapter.available():
        pytest.skip(f"optional backend unavailable: {backend}")
    expected = yat_embedding(canonical_embedding_case())
    actual = adapter.embedding_value_and_grad(
        canonical_embedding_case(), compiled=compiled
    )
    compare(
        f"{backend}:embedding-output",
        actual.output,
        expected.output,
        expected_dtype="float32",
        **TOLERANCE,
    )


@pytest.mark.parametrize("backend,reference", ADAPTERS.items())
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_embedding_attend_outputs_and_all_supported_gradients_match_float64_oracle(
    backend, reference, compiled
):
    adapter = load_adapter(reference)
    if not adapter.available():
        pytest.skip(f"optional backend unavailable: {backend}")
    case = canonical_embedding_attend_case()
    expected = yat_embedding_attend(case)
    actual = adapter.embedding_attend_value_and_grad(case, compiled=compiled)
    compare(
        f"{backend}:embed-attend-output",
        actual.output,
        expected.output,
        expected_dtype="float32",
        **TOLERANCE,
    )
    assert set(actual.gradients) == set(expected.gradients)
    for name, expected_gradient in expected.gradients.items():
        compare(
            f"{backend}:embed-attend-{name}",
            actual.gradients[name],
            expected_gradient,
            expected_dtype="float32",
            **TOLERANCE,
        )
    compare(
        f"{backend}:embedding-gradient",
        actual.gradients["embedding"],
        expected.gradients["embedding"],
        expected_dtype="float32",
        **TOLERANCE,
    )


@pytest.mark.parametrize("backend,reference", ADAPTERS.items())
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize(
    ("transpose", "case_factory", "oracle"),
    [
        (False, canonical_convolution_case, yat_conv1d),
        (True, canonical_transpose_convolution_case, yat_conv_transpose1d),
    ],
    ids=["convolution", "transpose-convolution"],
)
def test_convolution_outputs_and_all_gradients_match_float64_oracle(
    backend, reference, compiled, transpose, case_factory, oracle
):
    adapter = load_adapter(reference)
    if not adapter.available():
        pytest.skip(f"optional backend unavailable: {backend}")
    case = case_factory()
    expected = oracle(case)
    actual = adapter.convolution_value_and_grad(
        case, transpose=transpose, compiled=compiled
    )
    compare(
        f"{backend}:conv-output",
        actual.output,
        expected.output,
        expected_dtype="float32",
        **TOLERANCE,
    )
    assert set(actual.gradients) == set(expected.gradients)
    for name, expected_gradient in expected.gradients.items():
        compare(
            f"{backend}:conv-{name}",
            actual.gradients[name],
            expected_gradient,
            expected_dtype="float32",
            **TOLERANCE,
        )
