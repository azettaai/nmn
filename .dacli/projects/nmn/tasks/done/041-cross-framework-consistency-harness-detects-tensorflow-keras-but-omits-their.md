---
id: t-01M198F2J58HDS74WMQFWY7MT6
kind: task
created: 2026-08-30T11:56:49Z
created_by: a-root
owner: a-root
github:
  issue: 93
  repo: azettaai/nmn
github_acceptance_import:
  issue: 93
  body_digest: sha256:e1bfc0075f5f92c605723c0ef2a6f1a2a77411ab8fb3a0aded37a7bf07ca710e
  actor: a-root
  imported_at: 2026-08-30T11:56:49Z
---
# Cross-framework consistency harness detects TensorFlow/Keras but omits their executions
## Context
Adopted from GitHub issue #93.

## Summary

The cross-framework consistency tests mark TensorFlow and Keras as available, but run_all_frameworks does not execute them. Later comparisons call max on empty result sets and fail.

## Reproduction

With TensorFlow/Keras installed, tests/integration/test_cross_framework_consistency.py fails three cases because availability detection includes these frameworks while run_all_frameworks omits them.

## Acceptance criteria

- [ ] Execute every framework reported as available, or exclude unsupported frameworks consistently with an explicit reason.
- [ ] Never reduce an empty comparison set.
- [ ] Add regression coverage for Torch/JAX-only and TensorFlow/Keras-enabled environments.

## Acceptance
- [x] Execute every framework reported as available, or exclude unsupported frameworks consistently with an explicit reason.
- [x] Never reduce an empty comparison set.
- [x] Add regression coverage for Torch/JAX-only and TensorFlow/Keras-enabled environments.
## Log
- 2026-08-30T13:34:09Z accepted by a-root
- 2026-08-30T13:34:09Z verified by `KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_cli.py tests/integration/test_cross_framework.py tests/integration/test_cross_framework_consistency.py tests/integration/test_yat_nmn_parity.py` (exit 0) in branch codex/acceptance-b48bad at b48bad5 — proves that tree builds, not that the work is in trunk
- 2026-08-30T13:34:09Z deliverable: no dacli/041-cross-framework-consistency-harness-detects-tensorflow-keras-but-omits-their branch — nothing to check against master
- 2026-08-30T13:34:09Z completed by a-root
## Verification Evidence
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_cli.py tests/integration/test_cross_framework.py tests/integration/test_cross_framework_consistency.py tests/integration/test_yat_nmn_parity.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_cli.py tests/integration/test_cross_framework.py tests/integration/test_cross_framework_consistency.py tests/integration/test_yat_nmn_parity.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":28565,"artifact_hash":"sha256:02bb29654c02a9057b3597ad0e235d2781c545acc8ff2333cc9e715c74007037","verifier":"a-root","branch":"codex/acceptance-b48bad","commit_sha":"b48bad55d6ce07740f912d40e2472b63f944d305","tree_sha":"a9f1e227a6d1b17c4b29d80f5a2db8b63c377a6b","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_cli.py tests/integration/test_cross_framework.py tests/integration/test_cross_framework_consistency.py tests/integration/test_yat_nmn_parity.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_cli.py tests/integration/test_cross_framework.py tests/integration/test_cross_framework_consistency.py tests/integration/test_yat_nmn_parity.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":28239,"artifact_hash":"sha256:1dedfd5685a0f43aa1f5ff1c3a9849836e5799b78484f1a8090930347101b60c","verifier":"a-root","branch":"codex/acceptance-b48bad","commit_sha":"b48bad55d6ce07740f912d40e2472b63f944d305","tree_sha":"a9f1e227a6d1b17c4b29d80f5a2db8b63c377a6b","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"b48bad55d6ce07740f912d40e2472b63f944d305","observed_at":"2026-08-30T13:34:09.137353Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
