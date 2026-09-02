---
id: t-01M17A5GVV7NWQJ2VCSQF8DFT4
kind: task
created: 2026-08-29T17:48:04Z
created_by: a-root
owner: a-root
github:
  issue: 74
  repo: azettaai/nmn
github_acceptance_import:
  issue: 74
  body_digest: sha256:d64285548a9dff3400cdb0a9c0183783c60f08afbf701ec80d6d59bd78e1f8cd
  actor: a-root
  imported_at: 2026-08-29T17:48:04Z
estimate: "{optimistic: 2, probable: 3, pessimistic: 5}"
depends_on: "[t-01M17A5H5SGEX76S2V8VZZCH6E]"
---
# Keras YatConv1D accepts causal padding but cannot execute it
## Context
Adopted from GitHub issue #74.

## Summary

Keras `YatConv1D` accepts `padding="causal"` but cannot execute it.

## Reproduction

On the JAX backend, the call raises:

```text
Unrecognized padding type: expected VALID, SAME, or SAME_LOWER, got causal
```

The implementation bypasses the base `Conv1D` causal pre-padding helper and passes `"causal"` directly to `keras.ops.conv` for the dot product and patch norm.

## Acceptance criteria

- [ ] Pre-pad consistently for both computations, then use valid convolution.
- [ ] Test causality, shapes, stride/dilation validation, and both Keras backends.

## Acceptance
- [x] Pre-pad consistently for both computations, then use valid convolution.
- [x] Test causality, shapes, stride/dilation validation, and both Keras backends.
## Log
- 2026-08-29T18:02:29Z dependency edit by a-root (event 01M17AZWVREZ055CHQMHQSBSRK)
- 2026-08-30T11:38:04Z accepted by a-root
- 2026-08-30T11:38:04Z verified by `KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py` (exit 0) in branch codex/acceptance-2a209 at 2a209b9 — proves that tree builds, not that the work is in trunk
- 2026-08-30T11:38:04Z deliverable: no dacli/013-keras-yatconv1d-accepts-causal-padding-but-cannot-execute-it branch — nothing to check against master
- 2026-08-30T11:38:04Z completed by a-root
## Verification Evidence
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":30499,"artifact_hash":"sha256:9b288564bd3da4efe92d9b62f65761ac3fa97f4885b09c352c37ce07427ff214","verifier":"a-root","branch":"codex/acceptance-2a209","commit_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","tree_sha":"eaff49d53d9a200fbf47f760a58c36d9a739e76a","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":28440,"artifact_hash":"sha256:be5bc2807be947f14c8c1b32dd877dbe3d6cec99f505744f97f2773f9436518b","verifier":"a-root","branch":"codex/acceptance-2a209","commit_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","tree_sha":"eaff49d53d9a200fbf47f760a58c36d9a739e76a","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","observed_at":"2026-08-30T11:38:04.203256Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
