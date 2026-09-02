---
id: t-01M17A5G3RX1YVMWKVAPFYDJ2G
kind: task
created: 2026-08-29T17:48:04Z
created_by: a-root
owner: a-root
github:
  issue: 78
  repo: azettaai/nmn
github_acceptance_import:
  issue: 78
  body_digest: sha256:8ccb893892652c8c066864e669a38c721c9004bde9cb7612282d4174b7b9da50
  actor: a-root
  imported_at: 2026-08-29T17:48:04Z
estimate: "{optimistic: 1, probable: 2, pessimistic: 3}"
---
# The nmn Keras extra installs TensorFlow instead of declaring Keras 3
## Context
Adopted from GitHub issue #78.

## Summary

The `nmn[keras]` extra declares `tensorflow>=2.10.0` instead of the Keras 3 package.

NMN documents this backend as multi-backend Keras 3, and `nmn.keras` runs with `KERAS_BACKEND=jax` without TensorFlow. A separate `nmn[tf]` extra already exists.

## Impact

JAX/PyTorch Keras users are forced to install a large unrelated backend, while the actual `keras>=3` API dependency is not expressed directly.

## Acceptance criteria

- [ ] Declare an appropriate Keras 3 dependency for `nmn[keras]`.
- [ ] Keep TensorFlow isolated to `nmn[tf]` (and include it in `all` as desired).
- [ ] Add clean-environment install/import tests for Keras on JAX and TensorFlow.

## Acceptance
- [x] Declare an appropriate Keras 3 dependency for `nmn[keras]`.
- [x] Keep TensorFlow isolated to `nmn[tf]` (and include it in `all` as desired).
- [x] Add clean-environment install/import tests for Keras on JAX and TensorFlow.
## Log
- 2026-08-30T11:36:39Z accepted by a-root
- 2026-08-30T11:36:39Z verified by `KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py` (exit 0) in branch codex/acceptance-2a209 at 2a209b9 — proves that tree builds, not that the work is in trunk
- 2026-08-30T11:36:39Z deliverable: no dacli/009-the-nmn-keras-extra-installs-tensorflow-instead-of-declaring-keras-3 branch — nothing to check against master
- 2026-08-30T11:36:39Z completed by a-root
## Verification Evidence
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":50352,"artifact_hash":"sha256:28ccd5b04b491e7e0939d2ada8812102a15e56e2c61c00e99380633b45600b74","verifier":"a-root","branch":"codex/acceptance-2a209","commit_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","tree_sha":"eaff49d53d9a200fbf47f760a58c36d9a739e76a","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":35454,"artifact_hash":"sha256:798f8d10b728100cf9221a436042b7effecafb609913d7abd756150013437f8e","verifier":"a-root","branch":"codex/acceptance-2a209","commit_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","tree_sha":"eaff49d53d9a200fbf47f760a58c36d9a739e76a","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":28750,"artifact_hash":"sha256:f1e953387c4e79491ab247e59d647eaecd0f817ebf6e650c0f519a89e0eb3c50","verifier":"a-root","branch":"codex/acceptance-2a209","commit_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","tree_sha":"eaff49d53d9a200fbf47f760a58c36d9a739e76a","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","observed_at":"2026-08-30T11:36:39.193429Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
