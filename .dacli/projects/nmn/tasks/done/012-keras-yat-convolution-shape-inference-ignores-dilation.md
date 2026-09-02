---
id: t-01M17A5GPQZQZD23CPQMTFMFJ1
kind: task
created: 2026-08-29T17:48:04Z
created_by: a-root
owner: a-root
github:
  issue: 75
  repo: azettaai/nmn
github_acceptance_import:
  issue: 75
  body_digest: sha256:2500b919b86bb00cb03b3dc46696aee97a4515d6210c041ec3b576d12031783e
  actor: a-root
  imported_at: 2026-08-29T17:48:04Z
estimate: "{optimistic: 2, probable: 3, pessimistic: 5}"
depends_on: "[t-01M17A5GVV7NWQJ2VCSQF8DFT4]"
---
# Keras YAT convolution shape inference ignores dilation
## Context
Adopted from GitHub issue #75.

## Summary

All six Keras YAT convolution `compute_output_shape` implementations ignore dilation.

## Reproduction

With kernel size 3, dilation 2, valid padding:

- `YatConv1D` declares length 7 for input length 9 but returns 5.
- `YatConvTranspose1D` declares length 7 for input length 5 but returns 9.
- Equivalent mismatches occur on every spatial axis in 2D and 3D.

## Impact

Functional-model shape inference, tracing, summaries, and serialization can be inconsistent with runtime tensors.

## Acceptance criteria

- [ ] Use effective dilated kernel sizes or Keras's canonical conv shape utilities.
- [ ] Test forward/transpose, all dimensions, padding modes, strides, dilation, and unknown dimensions.

## Acceptance
- [x] Use effective dilated kernel sizes or Keras's canonical conv shape utilities.
- [x] Test forward/transpose, all dimensions, padding modes, strides, dilation, and unknown dimensions.
## Log
- 2026-08-29T18:02:29Z dependency edit by a-root (event 01M17AZX1T827SWRJ05Y4MZ5GZ)
- 2026-08-30T11:37:35Z accepted by a-root
- 2026-08-30T11:37:35Z verified by `KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py` (exit 0) in branch codex/acceptance-2a209 at 2a209b9 — proves that tree builds, not that the work is in trunk
- 2026-08-30T11:37:35Z deliverable: no dacli/012-keras-yat-convolution-shape-inference-ignores-dilation branch — nothing to check against master
- 2026-08-30T11:37:35Z completed by a-root
## Verification Evidence
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":29955,"artifact_hash":"sha256:92aa8f654ea2fed89ce7452c530e1bf6bf715bdb21a4b8262112bf135e6826c1","verifier":"a-root","branch":"codex/acceptance-2a209","commit_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","tree_sha":"eaff49d53d9a200fbf47f760a58c36d9a739e76a","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":27886,"artifact_hash":"sha256:31aef38eb6303fd39bc4f5e3a49e984ea19dbb09133db2fa00db63e2c9fb977e","verifier":"a-root","branch":"codex/acceptance-2a209","commit_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","tree_sha":"eaff49d53d9a200fbf47f760a58c36d9a739e76a","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","observed_at":"2026-08-30T11:37:35.512719Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
