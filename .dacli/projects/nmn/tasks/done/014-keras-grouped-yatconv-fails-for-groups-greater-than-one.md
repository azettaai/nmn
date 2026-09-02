---
id: t-01M17A5H5SGEX76S2V8VZZCH6E
kind: task
created: 2026-08-29T17:48:05Z
created_by: a-root
owner: a-root
github:
  issue: 73
  repo: azettaai/nmn
github_acceptance_import:
  issue: 73
  body_digest: sha256:52c2357fbe757f73a2a9b57c49ff1c39213e0c93a900335a9791328f532b8ca1
  actor: a-root
  imported_at: 2026-08-29T17:48:05Z
estimate: "{optimistic: 3, probable: 5, pessimistic: 8}"
depends_on: "[t-01M17A5NB28MX1Y4JQMG1XEJ4C]"
---
# Keras grouped YatConv fails for groups greater than one
## Context
Adopted from GitHub issue #73.

## Summary

Keras `YatConv1D/2D/3D` fail whenever `groups > 1`.

## Reproduction

With the JAX backend, input channels 4, filters 4, and groups 2, all dimensions raise:

```text
rhs output feature dimension size must be a multiple of feature_group_count,
but 1 is not a multiple of 2
```

The patch-norm helper kernel has one output channel while `ops.conv` infers grouped convolution.

## Acceptance criteria

- [ ] Produce one norm channel per group and repeat within each group.
- [ ] Test 1D/2D/3D grouped forwards and gradients on JAX and TensorFlow backends.

## Acceptance
- [x] Produce one norm channel per group and repeat within each group.
- [x] Test 1D/2D/3D grouped forwards and gradients on JAX and TensorFlow backends.
## Log
- 2026-08-29T18:02:28Z dependency edit by a-root (event 01M17AZWNKW12YZ9W6W4A2QMM8)
- 2026-08-30T11:38:32Z accepted by a-root
- 2026-08-30T11:38:32Z verified by `KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py` (exit 0) in branch codex/acceptance-2a209 at 2a209b9 — proves that tree builds, not that the work is in trunk
- 2026-08-30T11:38:32Z deliverable: no dacli/014-keras-grouped-yatconv-fails-for-groups-greater-than-one branch — nothing to check against master
- 2026-08-30T11:38:32Z completed by a-root
## Verification Evidence
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":29475,"artifact_hash":"sha256:7408ab6f5c1edd88d86fed421a82b11470c9ec8b89fc61277f0bbf4199a78de9","verifier":"a-root","branch":"codex/acceptance-2a209","commit_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","tree_sha":"eaff49d53d9a200fbf47f760a58c36d9a739e76a","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":28088,"artifact_hash":"sha256:8ad9a4a2682bd6c8d7bfc804d3b7cd5a2bf10edc2eb6b4ee2fb1cf4145015e64","verifier":"a-root","branch":"codex/acceptance-2a209","commit_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","tree_sha":"eaff49d53d9a200fbf47f760a58c36d9a739e76a","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","observed_at":"2026-08-30T11:38:32.558022Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
