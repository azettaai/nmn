"""Schema, documentation, and dependency-boundary contract tests."""

from __future__ import annotations

import copy
import json
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
    ],
)
def test_contract_validator_fails_closed(mutation, message):
    from nmn.conformance import validate_contract

    contract = copy.deepcopy(load_contract())
    mutation(contract)
    with pytest.raises(ContractError, match=message):
        validate_contract(contract)


def test_contract_loader_imports_no_optional_framework():
    program = """
import json, sys
from nmn.conformance import load_contract
load_contract()
print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in {
    'torch', 'tensorflow', 'keras', 'jax', 'flax', 'mlx'
})))
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []


def test_generated_support_document_is_current():
    expected = (ROOT / "docs" / "generated" / "conformance.md").read_text()
    assert expected == render_support_markdown()
