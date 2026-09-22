"""Central test ownership. Every collected test receives exactly one tier (#160)."""

from pathlib import Path

TIERS = ("unit", "backend", "conformance", "accelerator", "performance")


def tier_for(path: Path) -> str:
    parts = path.parts
    if "benchmarks" in parts:
        return "performance"
    if "test_mlx" in parts or "accelerators" in parts:
        return "accelerator"
    if "conformance" in parts or "integration" in parts:
        return "conformance"
    if any(
        p in {"test_torch", "test_nnx", "test_linen", "test_tf", "test_keras"}
        for p in parts
    ):
        return "backend"
    return "unit"
