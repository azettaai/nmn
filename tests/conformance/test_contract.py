"""Schema, documentation, and dependency-boundary contract tests."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nmn.conformance import ContractError, load_contract, render_support_markdown

ROOT = Path(__file__).resolve().parents[2]


def test_packaged_contract_is_valid_json_and_complete():
    contract = load_contract()
    assert contract["schema_version"] == 1
    assert len(contract["backends"]) == 6
    assert len(contract["operations"]) == 7
    assert all(
        set(backend["operations"]) == set(contract["operations"])
        for backend in contract["backends"].values()
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("tolerances"), "missing required keys"),
        (lambda value: value.__setitem__("schema_version", 2), "must be 1"),
        (
            lambda value: value["backends"]["torch"]["operations"].pop("ray"),
            "must declare every operation",
        ),
        (
            lambda value: value["backends"]["mlx"]["operations"]["dense"].__setitem__(
                "status", "maybe"
            ),
            "status is invalid",
        ),
        (
            lambda value: value["profiles"]["torch"]["dense"].__setitem__(
                "dtypes", "float32"
            ),
            "dtypes must be an object",
        ),
        (
            lambda value: value["profiles"]["torch"]["dense"].__setitem__("config", {}),
            "config must not be empty",
        ),
        (
            lambda value: value["profiles"]["tf"]["embed"]["dtypes"].__setitem__(
                "tested", ["float32"]
            ),
            "declared evidence must not claim test coverage",
        ),
        (
            lambda value: value["backends"]["torch"]["operations"]["dense"].__setitem__(
                "status", "unsupported"
            ),
            "unsupported capability must not claim support or tests",
        ),
    ],
)
def test_contract_validator_fails_closed(mutation, message):
    from nmn.conformance import validate_contract

    contract = copy.deepcopy(load_contract())
    mutation(contract)
    with pytest.raises(ContractError, match=message):
        validate_contract(contract)


def test_every_capability_has_an_exact_profile_cell():
    contract = load_contract()
    for backend_name, backend in contract["backends"].items():
        profiles = contract["profiles"][backend_name]
        assert set(profiles) == set(backend["operations"])
        for operation_name, profile in profiles.items():
            assert set(profile) == {
                "dtypes",
                "modes",
                "keras_engine",
                "config",
                "layout",
                "masking",
                "serialization",
                "evidence",
            }
            assert profile["dtypes"]["supported"]
            assert profile["modes"]["supported"]
            assert profile["config"]
            assert profile["serialization"]
            if backend["operations"][operation_name]["conformance"] == "declared":
                assert profile["dtypes"]["tested"] == []
                assert profile["modes"]["tested"] == []
                assert profile["evidence"]["tests"] == []


def test_contract_loader_imports_no_optional_framework():
    program = """
import json, sys
from nmn.conformance import load_contract
load_contract()
print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in {
    'torch', 'tensorflow', 'keras', 'jax', 'flax', 'mlx'
})))
"""
    environment = os.environ.copy()
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (source, environment.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(result.stdout) == []


@pytest.mark.parametrize("engine", ["jax", "torch"])
def test_keras_adapter_does_not_claim_unimplemented_engines(engine):
    program = """
from tests.conformance.adapters.keras import KerasAdapter
assert KerasAdapter.available() is False
"""
    environment = os.environ.copy()
    environment["KERAS_BACKEND"] = engine
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), source, environment.get("PYTHONPATH", "")) if part
    )
    subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_generated_support_document_is_current():
    expected = (ROOT / "docs" / "generated" / "conformance.md").read_text()
    assert expected == render_support_markdown()
