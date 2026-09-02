---
id: f-root-nmn-all-defeats-optional-backend-import-isolation-for-wildcard-imports
kind: note
note_kind: finding
created: 2026-08-22T21:52:10Z
created_by: a-root
about: "[[t-01M0N7B0WY37YQY22JWDA1B4GC]]"
severity: minor
---
# Root nmn __all__ defeats optional-backend import isolation for wildcard imports
Runtime-reproduced: PYTHONPATH=src python -c 'from nmn import *' eagerly attempts every name in __all__, importing optional backend packages; in this environment it fails in nmn.keras because TensorFlow is absent (and can later hit headless MLX). import nmn alone remains light, but advertising torch/nnx/keras/tf/linen/mlx in __all__ makes standard wildcard import require every optional dependency, contrary to the independent-backend contract.
