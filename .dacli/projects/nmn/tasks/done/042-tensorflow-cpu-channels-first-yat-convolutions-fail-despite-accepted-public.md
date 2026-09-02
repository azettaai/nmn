---
id: t-01M198F2NDWEJWXKK93JGK0M3K
kind: task
created: 2026-08-30T11:56:49Z
created_by: a-root
owner: a-root
github:
  issue: 92
  repo: azettaai/nmn
github_acceptance_import:
  issue: 92
  body_digest: sha256:75c651f37ff2cfec5fe706c13090492d3baeba37a4f3fac190cdb5c6ac827494
  actor: a-root
  imported_at: 2026-08-30T11:56:49Z
---
# TensorFlow CPU channels_first YAT convolutions fail despite accepted public format
## Context
Adopted from GitHub issue #92.

## Summary

Several public Keras/TensorFlow YAT convolution paths accept data_format=channels_first but call raw NCHW/NCDHW TensorFlow CPU operations that are unsupported.

## Reproduction

In TensorFlow 2.21 CPU integration tests, YatConv1D, YatConvTranspose1D, YatConv2D, YatConvTranspose2D, and YatConvTranspose3D fail for channels_first. Forward YatConv3D happens to work. The API and docs do not reject or qualify this format.

## Acceptance criteria

- [ ] Implement an internal transpose/fallback so every accepted channels_first 1D/2D/3D forward and transpose layer runs on TensorFlow CPU, or explicitly reject unsupported cases before raw backend execution with documented behavior.
- [ ] Add channels_first versus channels_last numerical and gradient parity tests.
- [ ] Cover eager and tf.function execution on CPU.

## Acceptance
- [x] Implement an internal transpose/fallback so every accepted channels_first 1D/2D/3D forward and transpose layer runs on TensorFlow CPU, or explicitly reject unsupported cases before raw backend execution with documented behavior.
- [x] Add channels_first versus channels_last numerical and gradient parity tests.
- [x] Cover eager and tf.function execution on CPU.
## Log
- 2026-08-30T13:04:04Z accepted by a-root
- 2026-08-30T13:04:04Z verified by `KERAS_BACKEND=tensorflow PYTHONPATH=src /private/tmp/nmn-tf-venv/bin/python -m pytest -q tests/test_keras/test_channels_first_tf_cpu.py` (exit 0) in branch codex/acceptance-9bc7 at 9bc7edc — proves that tree builds, not that the work is in trunk
- 2026-08-30T13:04:04Z deliverable: no dacli/042-tensorflow-cpu-channels-first-yat-convolutions-fail-despite-accepted-public branch — nothing to check against master
- 2026-08-30T13:04:04Z completed by a-root
## Verification Evidence
{"command":"KERAS_BACKEND=tensorflow PYTHONPATH=src /private/tmp/nmn-tf-venv/bin/python -m pytest -q tests/test_keras/test_channels_first_tf_cpu.py","argv":["sh","-c","KERAS_BACKEND=tensorflow PYTHONPATH=src /private/tmp/nmn-tf-venv/bin/python -m pytest -q tests/test_keras/test_channels_first_tf_cpu.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":3667,"artifact_hash":"sha256:eb8f6e01f71936c750e34bf3d29175ddef16498dfd883e489676b130137a41d3","verifier":"a-root","branch":"codex/acceptance-9bc7","commit_sha":"9bc7edc493d6b8b1b9397d5bdd138386e749e28a","tree_sha":"7537f0bd68b5a153da349ffda3c8e99474569f4f","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"KERAS_BACKEND=tensorflow PYTHONPATH=src /private/tmp/nmn-tf-venv/bin/python -m pytest -q tests/test_keras/test_channels_first_tf_cpu.py","argv":["sh","-c","KERAS_BACKEND=tensorflow PYTHONPATH=src /private/tmp/nmn-tf-venv/bin/python -m pytest -q tests/test_keras/test_channels_first_tf_cpu.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":3658,"artifact_hash":"sha256:dd8963a7b50ad2fbc3b0eb1586cfc0d7e8304b495a9edc6aff406ea0579c32b7","verifier":"a-root","branch":"codex/acceptance-9bc7","commit_sha":"9bc7edc493d6b8b1b9397d5bdd138386e749e28a","tree_sha":"7537f0bd68b5a153da349ffda3c8e99474569f4f","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"9bc7edc493d6b8b1b9397d5bdd138386e749e28a","observed_at":"2026-08-30T13:04:04.04031Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
