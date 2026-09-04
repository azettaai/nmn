"""Contract-generated dense forward parity across thin backend adapters."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nmn.conformance import load_contract
from tests.conformance.harness import compare, load_adapter
from tests.conformance.oracle import (
    canonical_dense_case,
    read_dense_fixture,
    yat_dense,
)

CONTRACT = load_contract()


def _cases():
    include_all_modes = os.environ.get("NMN_CONFORMANCE_ALL_MODES") == "1"
    for backend_name, backend in CONTRACT["backends"].items():
        modes = backend["execution"] if include_all_modes else ["eager"]
        for mode in modes:
            yield pytest.param(
                backend_name,
                backend,
                mode,
                id=f"{backend_name}-{mode}",
            )


@pytest.mark.parametrize("backend_name,backend,mode", list(_cases()))
def test_dense_forward_matches_float64_oracle(backend_name, backend, mode):
    adapter = load_adapter(backend["adapter"])
    if not adapter.available():
        platform = os.environ.get("NMN_CONFORMANCE_PLATFORM")
        required_here = backend["required_in_ci"] and backend["ci_platform"] == platform
        if required_here:
            pytest.fail(f"required {platform} backend is unavailable: {backend_name}")
        pytest.skip(f"optional backend unavailable: {backend_name}")

    fixture_path = os.environ.get("NMN_CONFORMANCE_FIXTURE")
    if backend_name == "mlx" and fixture_path:
        case, expected = read_dense_fixture(Path(fixture_path))
    else:
        case = canonical_dense_case(CONTRACT["canonical"]["random_seed"])
        expected = yat_dense(case)
    actual = adapter.dense(case, compiled=mode != "eager")
    tolerance = CONTRACT["tolerances"]["float32"]
    compare(backend_name, actual, expected.output, **tolerance)


@pytest.mark.parametrize("backend_name,backend,mode", list(_cases()))
def test_dense_output_and_all_gradients_match_float64_oracle(
    backend_name, backend, mode
):
    adapter = load_adapter(backend["adapter"])
    if not adapter.available():
        platform = os.environ.get("NMN_CONFORMANCE_PLATFORM")
        required_here = backend["required_in_ci"] and backend["ci_platform"] == platform
        if required_here:
            pytest.fail(f"required {platform} backend is unavailable: {backend_name}")
        pytest.skip(f"optional backend unavailable: {backend_name}")

    fixture_path = os.environ.get("NMN_CONFORMANCE_FIXTURE")
    if backend_name == "mlx" and fixture_path:
        case, expected = read_dense_fixture(Path(fixture_path))
    else:
        case = canonical_dense_case(CONTRACT["canonical"]["random_seed"])
        expected = yat_dense(case)
    actual = adapter.dense_value_and_grad(case, compiled=mode != "eager")
    tolerance = CONTRACT["tolerances"]["float32"]
    compare(backend_name, actual.output, expected.output, **tolerance)
    assert set(actual.gradients) == set(expected.gradients)
    for name, expected_gradient in expected.gradients.items():
        compare(
            f"{backend_name}:{name}",
            actual.gradients[name],
            expected_gradient,
            **tolerance,
        )


def test_every_adapter_reference_is_loadable_without_importing_its_framework():
    for backend in CONTRACT["backends"].values():
        adapter = load_adapter(backend["adapter"])
        assert callable(adapter.available)
        assert callable(adapter.dense)
        assert callable(adapter.dense_value_and_grad)


def test_every_declared_api_symbol_exists_when_backend_is_available():
    for backend_name, backend in CONTRACT["backends"].items():
        adapter = load_adapter(backend["adapter"])
        if not adapter.available():
            continue
        module = __import__(f"nmn.{backend_name}", fromlist=["nmn"])
        for capability in backend["operations"].values():
            for api_name in capability["api"].split("|"):
                assert hasattr(module, api_name.rsplit(".", 1)[1]), api_name
