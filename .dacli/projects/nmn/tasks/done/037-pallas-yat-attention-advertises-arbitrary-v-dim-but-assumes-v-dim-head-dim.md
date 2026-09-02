---
id: t-01M17A5P27MVPMRPA417QHC48J
kind: task
created: 2026-08-29T17:48:10Z
created_by: a-root
owner: a-root
github:
  issue: 50
  repo: azettaai/nmn
github_acceptance_import:
  issue: 50
  body_digest: sha256:0172300dae0b9fa117a9f3b2b63c4ce60f4bf42894fdbbd246a4ba471113d24b
  actor: a-root
  imported_at: 2026-08-29T17:48:10Z
estimate: "{optimistic: 3, probable: 5, pessimistic: 8}"
depends_on: "[t-01M17A5P8V2NMFAAKA4CAKS4A5]"
---
# Pallas YAT attention advertises arbitrary v_dim but assumes v_dim == head_dim
## Context
Adopted from GitHub issue #50.

## Summary

The public API documents values as `[..., kv_len, num_heads, v_dim]` and promises output shape `[..., q_len, num_heads, v_dim]`, but the Pallas implementation allocates and tiles V/output using Q/K `head_dim`. Passing a valid attention configuration where `v_dim != head_dim` fails.

## Affected code

- `src/nmn/nnx/layers/attention/pallas_yat_attention.py`
- Forward output shape/specs and both backward kernels

## Reproduction

On commit `4f0298d`, CPU Pallas interpret mode:

```python
import jax
from nmn.nnx.layers.attention.pallas_yat_attention import pallas_yat_l1_attention

q = jax.random.normal(jax.random.key(0), (1, 8, 2, 4))
k = jax.random.normal(jax.random.key(1), (1, 8, 2, 4))
v = jax.random.normal(jax.random.key(2), (1, 8, 2, 3))

pallas_yat_l1_attention(
    q, k, v, block_q=4, block_k=4, interpret=True
)
```

Observed:

```
TypeError: sub got incompatible shapes for broadcasting:
(1, 8, 2, 4), (1, 8, 2, 3)
```

The non-Pallas fused reference accepts this input and returns shape `(1, 8, 2, 3)`.

## Impact

The Pallas GPU/TPU implementation is not a drop-in replacement for standard attention when value projections use a different per-head width.

## Acceptance criteria

- [ ] Track `head_dim` and `v_dim` independently in kernel shapes and block specs, or explicitly reject/document `v_dim != head_dim`.
- [ ] Forward output has the documented `v_dim`.
- [ ] dV and upstream gradients have correct shapes and numerical parity with the fused reference.
- [ ] Tests cover at least one `v_dim < head_dim` and one `v_dim > head_dim` case.

## Acceptance
- [x] Track `head_dim` and `v_dim` independently in kernel shapes and block specs, or explicitly reject/document `v_dim != head_dim`.
- [x] Forward output has the documented `v_dim`.
- [x] dV and upstream gradients have correct shapes and numerical parity with the fused reference.
- [x] Tests cover at least one `v_dim < head_dim` and one `v_dim > head_dim` case.
## Log
- 2026-08-29T18:02:31Z dependency edit by a-root (event 01M17AZZK27M4C1JV9PTZMSS3H)
- 2026-08-30T09:09:57Z accepted by a-root
- 2026-08-30T09:09:57Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_pallas_yat_attention.py` (exit 0) in branch codex/acceptance-b88 at b88e5ed — proves that tree builds, not that the work is in trunk
- 2026-08-30T09:09:57Z deliverable: no dacli/037-pallas-yat-attention-advertises-arbitrary-v-dim-but-assumes-v-dim-head-dim branch — nothing to check against master
- 2026-08-30T09:09:57Z completed by a-root
## Verification Evidence
{"command":"cd /Users/tahabsn/conductor/workspaces/nmn/islamabad \u0026\u0026 PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_pallas_yat_attention.py","argv":["sh","-c","cd /Users/tahabsn/conductor/workspaces/nmn/islamabad \u0026\u0026 PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_pallas_yat_attention.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":26293,"artifact_hash":"sha256:7bdd48702ee0b80213eccb77f1c23a3cf5cb436d07aacfb24ce0f726b6799e3b","verifier":"a-root","branch":"dacli/040-add-bf16-native-and-mixed-precision-yatnmn-execution-modes","commit_sha":"4635a545a7096db0f04e81daa8f91db46c883a07","tree_sha":"05776a08a1f9a0373b63b9bd6378c03ed7fa03c1","runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_pallas_yat_attention.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_pallas_yat_attention.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":22155,"artifact_hash":"sha256:7e32003c00e8895f6f0a023b125d0556b2cc8f67a21916eab41a737611f0a860","verifier":"a-root","branch":"codex/acceptance-b88","commit_sha":"b88e5eda007cb2baa1205d9e803d4aff81665aad","tree_sha":"fb080c6983893af1e0e3e4bc5eacb87a42d2d970","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"b88e5eda007cb2baa1205d9e803d4aff81665aad","observed_at":"2026-08-30T09:09:57.643972Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
