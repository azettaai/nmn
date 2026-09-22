"""Mechanical convolution refactor compatibility and tier coverage (#152, #160)."""

import ast
import json
from pathlib import Path

from tests._tiers import TIERS, tier_for

ROOT = Path(__file__).resolve().parents[1]


def constructors(source):
    result = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.ClassDef) or not node.name.startswith("YatConv"):
            continue
        init = next(
            (
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            ),
            None,
        )
        if init:
            args = init.args
            for arg in (
                args.posonlyargs
                + args.args
                + args.kwonlyargs
                + ([args.kwarg] if args.kwarg else [])
            ):
                arg.annotation = None
            result[node.name] = ast.unparse(args)
        else:
            result[node.name] = [
                (n.target.id, ast.unparse(n.value) if n.value else None)
                for n in node.body
                if isinstance(n, ast.AnnAssign) and not n.target.id.startswith("_")
            ]
    return json.loads(json.dumps(result))


def test_convolution_constructor_defaults_remain_compatible():
    baseline = json.loads((ROOT / "tests/fixtures/convolution_api.json").read_text())
    for backend, expected in baseline.items():
        assert (
            constructors((ROOT / f"src/nmn/{backend}/conv.py").read_text()) == expected
        )


def test_every_test_module_has_one_known_tier():
    for path in (ROOT / "tests").rglob("test_*.py"):
        assert tier_for(path) in TIERS


def test_convolution_private_keras_imports_are_isolated():
    for path in (ROOT / "src/nmn/keras").glob("*.py"):
        if path.name == "_compat.py":
            continue
        assert "from keras.src" not in path.read_text(), path


def test_regression_split_retains_all_original_test_names():
    baseline = json.loads((ROOT / "tests/fixtures/regression_names.json").read_text())
    for backend, names in baseline.items():
        actual = []
        for path in (ROOT / f"tests/test_{backend}").glob("test_regression_*.py"):
            actual += [
                n.name
                for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
            ]
        assert sorted(actual) == names


def test_legacy_torch_base_warns_without_changing_exports():
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import warnings; warnings.simplefilter('error', DeprecationWarning); import nmn.torch.base",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    # This is a dependency-light policy test: import may report missing Torch.
    if "optional dependency 'torch'" in result.stderr:
        return
    assert result.returncode != 0
    assert "nmn.torch.base is deprecated" in result.stderr
