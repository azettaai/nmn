"""Import-light access to NMN's cross-framework semantic contract."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any, Mapping

BACKENDS = ("torch", "nnx", "linen", "tf", "keras", "mlx")
OPERATIONS = (
    "dense",
    "embed",
    "convolution",
    "transpose_convolution",
    "attention",
    "may",
    "ray",
)
EXECUTION_MODES = {"eager", "compile", "jit", "function", "compiled_call"}
SUPPORT_STATES = {"supported", "partial", "unsupported"}
CONFORMANCE_STATES = {"oracle", "fixture", "declared"}


class ContractError(ValueError):
    """Raised when a conformance manifest violates the public schema."""


def load_contract() -> dict[str, Any]:
    """Load and validate the packaged cross-framework contract.

    This function uses only the Python standard library and never imports an ML
    framework, so tooling can inspect the matrix from a base ``nmn`` install.
    """
    resource = files("nmn").joinpath("conformance_manifest.json")
    contract = json.loads(resource.read_text(encoding="utf-8"))
    validate_contract(contract)
    return contract


def _require_keys(value: Mapping[str, Any], keys: set[str], path: str) -> None:
    missing = sorted(keys - value.keys())
    if missing:
        raise ContractError(f"{path} is missing required keys: {', '.join(missing)}")


def validate_contract(contract: Mapping[str, Any]) -> None:
    """Validate a manifest without importing optional backend dependencies."""
    _require_keys(
        contract,
        {
            "schema_version",
            "contract_version",
            "canonical",
            "tolerances",
            "operations",
            "backends",
        },
        "contract",
    )
    if contract["schema_version"] != 1:
        raise ContractError("contract.schema_version must be 1")

    operations = contract["operations"]
    backends = contract["backends"]
    if tuple(operations) != OPERATIONS:
        raise ContractError(f"contract.operations must be exactly {OPERATIONS!r}")
    if tuple(backends) != BACKENDS:
        raise ContractError(f"contract.backends must be exactly {BACKENDS!r}")

    for dtype in ("float64", "float32", "bfloat16", "float16"):
        tolerance = contract["tolerances"].get(dtype)
        if not isinstance(tolerance, Mapping):
            raise ContractError(f"contract.tolerances.{dtype} must be an object")
        _require_keys(tolerance, {"rtol", "atol"}, f"contract.tolerances.{dtype}")
        if tolerance["rtol"] < 0 or tolerance["atol"] < 0:
            raise ContractError(f"contract.tolerances.{dtype} must be nonnegative")

    for operation_name, operation in operations.items():
        _require_keys(
            operation,
            {"oracle", "outputs", "gradients", "masking", "layout"},
            f"contract.operations.{operation_name}",
        )
        if not operation["outputs"]:
            raise ContractError(
                f"contract.operations.{operation_name}.outputs is empty"
            )

    for backend_name, backend in backends.items():
        path = f"contract.backends.{backend_name}"
        _require_keys(
            backend,
            {
                "display_name",
                "adapter",
                "ci_platform",
                "required_in_ci",
                "execution",
                "dtypes",
                "serialization",
                "native_layout",
                "masking",
                "operations",
            },
            path,
        )
        unknown_modes = set(backend["execution"]) - EXECUTION_MODES
        if unknown_modes:
            raise ContractError(
                f"{path}.execution has unknown modes: {sorted(unknown_modes)}"
            )
        if set(backend["operations"]) != set(OPERATIONS):
            raise ContractError(f"{path}.operations must declare every operation")
        if set(backend["masking"]) != {"attention", "may", "ray"}:
            raise ContractError(f"{path}.masking must declare attention, may, and ray")
        for operation_name, capability in backend["operations"].items():
            capability_path = f"{path}.operations.{operation_name}"
            _require_keys(capability, {"status", "conformance", "api"}, capability_path)
            if capability["status"] not in SUPPORT_STATES:
                raise ContractError(f"{capability_path}.status is invalid")
            if capability["conformance"] not in CONFORMANCE_STATES:
                raise ContractError(f"{capability_path}.conformance is invalid")
            if not isinstance(capability["api"], str):
                raise ContractError(f"{capability_path}.api must be a string")
            api_names = capability["api"].split("|")
            if not api_names or any(
                not name.startswith(f"nmn.{backend_name}.") for name in api_names
            ):
                raise ContractError(
                    f"{capability_path}.api must contain fully qualified "
                    f"nmn.{backend_name} symbols"
                )


def render_support_markdown(contract: Mapping[str, Any] | None = None) -> str:
    """Render the human support table from the canonical manifest."""
    if contract is None:
        contract = load_contract()
    lines = [
        "<!-- Generated by `python -m nmn.conformance`; do not edit by hand. -->",
        "# Cross-framework conformance contract",
        "",
        f"Contract version: `{contract['contract_version']}`.",
        "",
        "| Backend | CI | Execution | Dtypes | Serialization | Linear masks |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for backend in contract["backends"].values():
        required = "required" if backend["required_in_ci"] else "optional"
        lines.append(
            f"| {backend['display_name']} | {backend['ci_platform']} ({required}) | "
            f"{', '.join(backend['execution'])} | {', '.join(backend['dtypes'])} | "
            f"{', '.join(backend['serialization'])} | "
            f"MAY: {', '.join(backend['masking']['may'])}; "
            f"RAY: {', '.join(backend['masking']['ray'])} |"
        )
    lines.extend(["", "## Operation coverage", ""])
    headings = [contract["backends"][name]["display_name"] for name in BACKENDS]
    lines.append("| Operation | " + " | ".join(headings) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in headings) + " |")
    for operation_name in OPERATIONS:
        cells = []
        for backend_name in BACKENDS:
            capability = contract["backends"][backend_name]["operations"][
                operation_name
            ]
            cells.append(f"{capability['status']} / {capability['conformance']}")
        lines.append(f"| {operation_name} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Enforced tolerances", ""])
    lines.append("| Dtype | rtol | atol |")
    lines.append("| --- | ---: | ---: |")
    for dtype, tolerance in contract["tolerances"].items():
        lines.append(
            f"| {dtype} | {tolerance['rtol']:.12g} | {tolerance['atol']:.12g} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Print generated support documentation to stdout."""
    print(render_support_markdown(), end="")


if __name__ == "__main__":
    main()
