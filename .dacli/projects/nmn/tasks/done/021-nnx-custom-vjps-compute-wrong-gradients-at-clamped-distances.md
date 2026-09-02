---
id: t-01M17A5KA5PQ0SP0KYRM1S3HBR
kind: task
created: 2026-08-29T17:48:07Z
created_by: a-root
owner: a-root
github:
  issue: 66
  repo: azettaai/nmn
github_acceptance_import:
  issue: 66
  body_digest: sha256:1c82bc1a81c1f60298167216604cfc58a6519a4180bb4ba9fca97036c85d2c50
  actor: a-root
  imported_at: 2026-08-29T17:48:07Z
estimate: "{optimistic: 3, probable: 5, pessimistic: 8}"
depends_on: "[t-01M17A5PJABDE71M0YXR39TSXA]"
---
# NNX custom VJPs compute wrong gradients at clamped distances
## Context
Adopted from GitHub issue #66.

## Summary

NNX optimized custom VJPs differentiate the raw reconstructed distance even when the forward `maximum(raw_distance, 0)` clamp is active.

Affected paths:

- optimized fused fp32 dense `YatNMN`
- fused L1 YAT attention

## Reproduction

For width-64 float32 dense input equal to its kernel (seed 2, scale 10), raw distance is `-0.0009765625`. Standard autodiff gives input-gradient norm `1.12576421888e11`; the fused VJP returns exactly zero while forwards match.

For fused L1 attention with nearly colliding keys, a seed search found up to 6.8% relative gradient error with equal forwards.

## Root cause

The analytic backward always propagates `g_dist` through `q² + k² - 2qk`; it omits the derivative mask of the clamp.

## Acceptance criteria

- [ ] Match the clamp subgradient used by JAX in both custom VJPs.
- [ ] Add forward and all-operand gradient parity at positive, zero, and negative reconstructed distances.

## Acceptance
- [x] Match the clamp subgradient used by JAX in both custom VJPs.
- [x] Add forward and all-operand gradient parity at positive, zero, and negative reconstructed distances.
## Log
- 2026-08-29T18:02:30Z dependency edit by a-root (event 01M17AZY4GNMH631EKDTM6FST8)
- 2026-08-30T09:11:57Z accepted by a-root
- 2026-08-30T09:11:57Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'fused_l1'` (exit 0) in branch codex/acceptance-b88 at b88e5ed — proves that tree builds, not that the work is in trunk
- 2026-08-30T09:11:57Z deliverable: no dacli/021-nnx-custom-vjps-compute-wrong-gradients-at-clamped-distances branch — nothing to check against master
- 2026-08-30T09:11:57Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'fused_l1'","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'fused_l1'"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":16457,"artifact_hash":"sha256:6402e4244f325fbdb5ece276651147b336b2f44b72614ce74a9259280d1f3902","verifier":"a-root","branch":"codex/acceptance-b88","commit_sha":"b88e5eda007cb2baa1205d9e803d4aff81665aad","tree_sha":"fb080c6983893af1e0e3e4bc5eacb87a42d2d970","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'fused_l1'","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'fused_l1'"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":17019,"artifact_hash":"sha256:600fc5f6c204cb6c4f90e32450f86608cb1e238b98b4db9a2272ecfbd411f2e0","verifier":"a-root","branch":"codex/acceptance-b88","commit_sha":"b88e5eda007cb2baa1205d9e803d4aff81665aad","tree_sha":"fb080c6983893af1e0e3e4bc5eacb87a42d2d970","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"b88e5eda007cb2baa1205d9e803d4aff81665aad","observed_at":"2026-08-30T09:11:57.090427Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
