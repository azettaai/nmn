"""Numerical tolerance policy shared by fixtures and conformance tests.

Conformance tolerances come from the packaged manifest. Device names are
explicit even where bounds agree, so unsupported combinations fail closed.
Feature-specific overflow/underflow tests may assert tighter local properties;
they are not evidence for broader public conformance claims.
"""

from nmn.conformance import load_contract

CONTRACT = load_contract()
DEVICES = {"cpu", "cuda", "tpu", "metal"}
LEGACY_REFERENCE = {"rtol": 1e-4, "atol": 1e-4}


def tolerance(dtype: str, operation: str, device: str = "cpu") -> dict[str, float]:
    """Return declared bounds; this does not assert that a device is tested."""
    if device not in DEVICES:
        raise ValueError(f"Unknown device: {device}")
    if operation not in CONTRACT["operations"]:
        raise ValueError(f"Unknown operation: {operation}")
    values = CONTRACT["tolerances"][dtype]
    return {"rtol": float(values["rtol"]), "atol": float(values["atol"])}
