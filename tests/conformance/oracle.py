"""NumPy float64 semantic oracles and canonical fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DenseCase:
    """Canonical backend-independent dense YAT operands."""

    inputs: np.ndarray
    kernel: np.ndarray
    bias: np.ndarray
    alpha: np.ndarray
    epsilon: np.ndarray
    cotangent: np.ndarray


@dataclass(frozen=True)
class OracleResult:
    """A forward result and gradients of its cotangent projection."""

    output: np.ndarray
    gradients: dict[str, np.ndarray]


def canonical_dense_case(seed: int = 2026) -> DenseCase:
    """Return deterministic, non-degenerate float64 conformance operands."""
    rng = np.random.default_rng(seed)
    return DenseCase(
        inputs=rng.normal(size=(3, 5)),
        kernel=rng.normal(size=(5, 4)),
        bias=rng.normal(size=(4,)),
        alpha=np.asarray(0.8),
        epsilon=np.asarray(2e-3),
        cotangent=rng.normal(size=(3, 4)),
    )


def yat_dense(case: DenseCase) -> OracleResult:
    """Evaluate canonical YAT dense semantics and all relevant gradients.

    The logical kernel layout is always ``[input_features, output_features]``.
    Bias is inside the squared affine term, distance is clamped at zero, and
    alpha multiplies the ratio.
    """
    x = np.asarray(case.inputs, dtype=np.float64)
    kernel = np.asarray(case.kernel, dtype=np.float64)
    bias = np.asarray(case.bias, dtype=np.float64)
    alpha = np.asarray(case.alpha, dtype=np.float64)
    epsilon = np.asarray(case.epsilon, dtype=np.float64)
    cotangent = np.asarray(case.cotangent, dtype=np.float64)

    dot = x @ kernel
    raw_distance = (
        np.sum(x * x, axis=-1, keepdims=True)
        + np.sum(kernel * kernel, axis=0, keepdims=True)
        - 2.0 * dot
    )
    distance = np.maximum(raw_distance, 0.0)
    affine = dot + bias
    denominator = distance + epsilon
    ratio = affine * affine / denominator
    output = ratio * alpha

    affine_bar = cotangent * alpha * 2.0 * affine / denominator
    distance_bar = -cotangent * alpha * affine * affine / (denominator * denominator)
    distance_bar = distance_bar * (raw_distance > 0.0)
    input_gradient = affine_bar @ kernel.T + np.sum(
        2.0 * distance_bar[..., None] * (x[:, None, :] - kernel.T[None, ...]),
        axis=1,
    )
    kernel_gradient = (
        x.T @ affine_bar
        + np.sum(
            2.0 * distance_bar[..., None] * (kernel.T[None, ...] - x[:, None, :]),
            axis=0,
        ).T
    )
    gradients = {
        "input": input_gradient,
        "kernel": kernel_gradient,
        "bias": np.sum(affine_bar, axis=0),
        "alpha": np.asarray(np.sum(cotangent * ratio)),
        "epsilon": np.asarray(
            np.sum(-cotangent * alpha * affine * affine / (denominator * denominator))
        ),
    }
    return OracleResult(output=output, gradients=gradients)


def write_dense_fixture(path: str | Path, case: DenseCase | None = None) -> None:
    """Serialize the canonical oracle fixture consumed by Apple MLX CI."""
    if case is None:
        case = canonical_dense_case()
    result = yat_dense(case)
    arrays = {
        "inputs": case.inputs,
        "kernel": case.kernel,
        "bias": case.bias,
        "alpha": case.alpha,
        "epsilon": case.epsilon,
        "cotangent": case.cotangent,
        "output": result.output,
    }
    arrays.update(
        {f"gradient_{name}": value for name, value in result.gradients.items()}
    )
    np.savez(path, **arrays)


def read_dense_fixture(path: str | Path) -> tuple[DenseCase, OracleResult]:
    """Load a canonical fixture without permitting pickled objects."""
    with np.load(path, allow_pickle=False) as fixture:
        case = DenseCase(
            inputs=fixture["inputs"],
            kernel=fixture["kernel"],
            bias=fixture["bias"],
            alpha=fixture["alpha"],
            epsilon=fixture["epsilon"],
            cotangent=fixture["cotangent"],
        )
        gradients = {
            name: fixture[f"gradient_{name}"]
            for name in ("input", "kernel", "bias", "alpha", "epsilon")
        }
        result = OracleResult(output=fixture["output"], gradients=gradients)
    return case, result
