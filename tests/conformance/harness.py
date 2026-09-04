"""Dependency-light adapter loading and conformance comparison."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tests.conformance.oracle import DenseCase, OracleResult


class Adapter(Protocol):
    """Thin backend boundary used by generated conformance cases."""

    @staticmethod
    def available() -> bool:
        """Return whether the backend runtime can be used."""
        ...

    @staticmethod
    def dense(case: DenseCase, *, compiled: bool = False) -> np.ndarray:
        """Run the canonical dense case."""
        ...

    @staticmethod
    def dense_value_and_grad(
        case: DenseCase, *, compiled: bool = False
    ) -> OracleResult:
        """Run dense and differentiate its canonical cotangent projection."""
        ...


@dataclass(frozen=True)
class Comparison:
    """Measured difference between a backend and the canonical oracle."""

    backend: str
    max_absolute_error: float
    max_relative_error: float


def load_adapter(reference: str) -> type[Adapter]:
    """Load ``package.module:Class`` only when a backend is exercised."""
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(f"invalid adapter reference: {reference!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def compare(
    backend: str,
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> Comparison:
    """Assert parity and return stable error metrics for diagnostics."""
    actual64 = np.asarray(actual, dtype=np.float64)
    expected64 = np.asarray(expected, dtype=np.float64)
    np.testing.assert_allclose(actual64, expected64, rtol=rtol, atol=atol)
    absolute = np.abs(actual64 - expected64)
    relative = absolute / np.maximum(np.abs(expected64), np.finfo(np.float64).tiny)
    return Comparison(backend, float(np.max(absolute)), float(np.max(relative)))
