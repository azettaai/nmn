---
id: t-01M198F2RNT4KH38F96P5GMDXY
kind: task
created: 2026-08-30T11:56:49Z
created_by: a-root
owner: a-root
github:
  issue: 91
  repo: azettaai/nmn
github_acceptance_import:
  issue: 91
  body_digest: sha256:2629b16cf6a1a88037401ede22e61784412413932aef6faf2b746b6b06383506
  actor: a-root
  imported_at: 2026-08-30T11:56:49Z
---
# Test harness can hard-abort while probing installed but unusable MLX
## Context
Adopted from GitHub issue #91.

## Summary

The test suite imports MLX in-process to detect availability. On an arm64 host where MLX is installed but Metal cannot initialize, the native runtime aborts the interpreter (exit 134), which cannot be caught by Python exception handling.

## Reproduction

PYTHONPATH=src python -m pytest -q tests/integration aborts during collection in tests/integration/test_cross_framework.py. Running tests/test_cli.py likewise aborts in its MLX probe. The CLI doctor command already uses subprocess isolation for this class of failure.

## Acceptance criteria

- [ ] Probe MLX availability in a subprocess anywhere the test harness may import it conditionally.
- [ ] Treat a nonzero or signaled probe as unavailable without aborting the parent pytest process.
- [ ] Add regression tests for successful import, Python exception, and native-process failure.

## Acceptance
- [x] Probe MLX availability in a subprocess anywhere the test harness may import it conditionally.
- [x] Treat a nonzero or signaled probe as unavailable without aborting the parent pytest process.
- [x] Add regression tests for successful import, Python exception, and native-process failure.
## Log
- 2026-08-30T13:35:06Z accepted by a-root
- 2026-08-30T13:35:06Z verified by `KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_cli.py tests/integration/test_cross_framework.py tests/integration/test_cross_framework_consistency.py tests/integration/test_yat_nmn_parity.py` (exit 0) in branch codex/acceptance-b48bad at b48bad5 — proves that tree builds, not that the work is in trunk
- 2026-08-30T13:35:06Z deliverable: no dacli/043-test-harness-can-hard-abort-while-probing-installed-but-unusable-mlx branch — nothing to check against master
- 2026-08-30T13:35:06Z completed by a-root
## Verification Evidence
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_cli.py tests/integration/test_cross_framework.py tests/integration/test_cross_framework_consistency.py tests/integration/test_yat_nmn_parity.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_cli.py tests/integration/test_cross_framework.py tests/integration/test_cross_framework_consistency.py tests/integration/test_yat_nmn_parity.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":28314,"artifact_hash":"sha256:ab2078e37865e3c909f1abdbbd48c002ed179b55f740cd40dbf73bbd306f5e4c","verifier":"a-root","branch":"codex/acceptance-b48bad","commit_sha":"b48bad55d6ce07740f912d40e2472b63f944d305","tree_sha":"a9f1e227a6d1b17c4b29d80f5a2db8b63c377a6b","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_cli.py tests/integration/test_cross_framework.py tests/integration/test_cross_framework_consistency.py tests/integration/test_yat_nmn_parity.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_cli.py tests/integration/test_cross_framework.py tests/integration/test_cross_framework_consistency.py tests/integration/test_yat_nmn_parity.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":28362,"artifact_hash":"sha256:ab2078e37865e3c909f1abdbbd48c002ed179b55f740cd40dbf73bbd306f5e4c","verifier":"a-root","branch":"codex/acceptance-b48bad","commit_sha":"b48bad55d6ce07740f912d40e2472b63f944d305","tree_sha":"a9f1e227a6d1b17c4b29d80f5a2db8b63c377a6b","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"b48bad55d6ce07740f912d40e2472b63f944d305","observed_at":"2026-08-30T13:35:06.221459Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
