---
id: t-01M17A5GHQ93F41NTCKEGMVW9E
kind: task
created: 2026-08-29T17:48:04Z
created_by: a-root
owner: a-root
github:
  issue: 76
  repo: azettaai/nmn
github_acceptance_import:
  issue: 76
  body_digest: sha256:95f4ca3301a08bef9761f2dbe553372c15860eb136914eec273ba02e6f791361
  actor: a-root
  imported_at: 2026-08-29T17:48:04Z
estimate: "{optimistic: 3, probable: 5, pessimistic: 8}"
depends_on: "[t-01M17A5GPQZQZD23CPQMTFMFJ1]"
---
# Keras convolution kernel-bank auto-expansion cannot resize variables
## Context
Adopted from GitHub issue #76.

## Summary

Keras YAT convolution kernel-bank auto-expansion cannot resize the existing variable.

## Reproduction

Create a tied bank with 3 filters, then request 5 filters using the same bank id. Forward and transpose 1D layers raise:

```text
The shape of the target variable and the shape of the target value in
variable.assign(value) must match
```

The same expansion code is duplicated in 1D/2D/3D forward and transpose layers. Keras variables have fixed shapes and cannot be expanded by `assign`.

## Acceptance criteria

- [ ] Implement expansion without illegal variable resizing, or reject it explicitly before mutation.
- [ ] Preserve existing values, tracking, serialization, and optimizer state.
- [ ] Test all six classes and multiple consumers.

## Acceptance
- [x] Implement expansion without illegal variable resizing, or reject it explicitly before mutation.
- [x] Preserve existing values, tracking, serialization, and optimizer state.
- [x] Test all six classes and multiple consumers.
## Log
- 2026-08-29T18:02:29Z dependency edit by a-root (event 01M17AZX862TRRP84YWGM9FN17)
- 2026-08-30T11:37:07Z accepted by a-root
- 2026-08-30T11:37:07Z verified by `KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py` (exit 0) in branch codex/acceptance-2a209 at 2a209b9 — proves that tree builds, not that the work is in trunk
- 2026-08-30T11:37:07Z deliverable: no dacli/011-keras-convolution-kernel-bank-auto-expansion-cannot-resize-variables branch — nothing to check against master
- 2026-08-30T11:37:07Z completed by a-root
## Verification Evidence
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":30281,"artifact_hash":"sha256:26e077acee0969d24b47c1bf49c2c479ae85dcd692479cfbcac616a1f94b9413","verifier":"a-root","branch":"codex/acceptance-2a209","commit_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","tree_sha":"eaff49d53d9a200fbf47f760a58c36d9a739e76a","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":27929,"artifact_hash":"sha256:08dedfa7f74a41d73c305caa24b86470124cac0cf2219754775bafb83c2ec078","verifier":"a-root","branch":"codex/acceptance-2a209","commit_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","tree_sha":"eaff49d53d9a200fbf47f760a58c36d9a739e76a","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","observed_at":"2026-08-30T11:37:07.372232Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
