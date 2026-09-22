# Test suite organization

Every collected test has exactly one tier marker assigned by `tests/_tiers.py`:

| Marker | Ownership | Command |
| --- | --- | --- |
| `unit` | Root dependency-light tests | `python -m pytest -m unit` |
| `backend` | Native framework behavior and serialization | `python -m pytest -m backend --require-backend torch` |
| `conformance` | NumPy oracle, canonical fixtures, integration contracts | `python -m pytest -m conformance` |
| `accelerator` | MLX Metal and explicit device test directories | `python -m pytest -m accelerator --require-backend mlx` |
| `performance` | Explicit benchmark programs | `make benchmark-attention` |

Use `python -m pytest -m unit` for the fast local loop. The complete available
matrix is `python -m pytest`; optional frameworks may skip locally. CI passes
`--require-backend` so missing installations cannot make a required job green.
Conformance CI separately requires the manifest's complete platform backend set.
Native TPU/CUDA checks remain external gates, not claimed by CPU interpret runs.

The suite is organized by the boundary it verifies:

- `test_<backend>/` — backend-specific unit and regression tests;
- `conformance/` — manifest-driven NumPy-oracle and fixture conformance;
- `integration/` — package-level behavior spanning multiple components;
- `test_cli.py` — import-light command-line behavior;
- `test_collection_policy.py` — optional-backend and collection isolation;
- `test_documentation_policy.py` — installation and release-metadata invariants;
- `test_workflow_policy.py` — CI/CD configuration invariants.

Exploratory programs, reports, and long-running benchmarks belong in the
repository-level `benchmarks/` directory. Files below `tests/` must be
deterministic, assertion-based, and safe to import during collection.

Useful commands:

```bash
python -m pytest -q -m "not slow"
python -m pytest tests/test_nnx -q
python -m pytest tests/conformance -q
python -m pytest tests/integration -q
python -m pytest \
  tests/test_workflow_policy.py \
  tests/test_collection_policy.py \
  tests/test_documentation_policy.py -q
mypy --no-error-summary
```

MyPy discovers the supported package surface from `[tool.mypy]` in
`pyproject.toml`. New Python modules below `src/nmn/` are checked automatically.
There are no package exclusions: every Python module, including examples and
all optional-backend implementations, participates in the same CI type check.
MyPy skips recursively checking imported dependencies so each package module
owns its errors consistently across the supported backend-version matrix.

Tests for unavailable optional backends are skipped before importing that
backend. Keep new optional-backend imports inside their backend tree or guarded
with `pytest.importorskip`.

## Benchmark policy

Wall-clock measurements are not correctness tests and are never part of the
default pytest or coverage runs. Run the explicit JAX attention benchmark with:

```bash
make benchmark-attention
```

The same command is available through the manually dispatched **Attention
Benchmarks** workflow. A run fails if any required implementation raises or
does not produce samples. Its JSON artifact records hardware and dependency
versions, the untimed warmup used to exclude lazy compilation, synchronization,
raw timing samples, and descriptive statistics. Compare artifacts only from
equivalent hardware and protocol metadata; timing ratios are reports, not merge
gates.

## Dependency and accelerator policy

Every pull request runs the latest supported CPU suites for JAX/Flax, PyTorch,
TensorFlow, and all three Keras backends. The declared lower bounds for Torch,
TensorFlow, Keras, and MLX receive a weekly representative runtime smoke test;
JAX/Flax's lower bound runs its full suite on every pull request because it is
the reference implementation. Dependabot keeps both mutable dependency ranges
and the immutable GitHub Action commit pins current.

Continuous accelerator coverage is intentionally explicit:

- MLX runs its full suite, fused Metal kernel gradients, and transpose-kernel
  parity on a real Apple Silicon GPU for every pull request;
- PyTorch, TensorFlow, Keras, and ordinary JAX tests run on CPU;
- Pallas kernels run their independent numerical oracle and BlockSpec legality
  checks in CPU interpret mode;
- native TPU Mosaic and CUDA execution remain external validation gates and
  are not represented as continuously tested in compatibility claims.

## Numerical tolerances and regression ownership

`tests/tolerances.py` is the shared dtype/device/operation lookup. Conformance
bounds come directly from `src/nmn/conformance_manifest.json`; the lookup does
not imply that every device is tested. Legacy reference defaults are named
separately. Precision stress tests retain explicit local bounds for their
specific overflow, collision, and gradient properties. Those bounds must not
be quoted as general cross-framework tolerance claims.

Torch and Keras issue regressions now live in `test_regression_*.py` grouped by
attention, convolution, kernel banks, precision, and serialization/device behavior.
The original test names are retained and checked against a migration inventory.
Common runners and conversions live in `_regression_support.py`; canonical
cross-framework adapters live only in `tests/conformance/adapters/`.
