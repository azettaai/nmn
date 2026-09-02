---
id: t-01M198F2VRP3P3SCS4S3VV69P8
kind: task
created: 2026-08-30T11:56:49Z
created_by: a-root
owner: a-root
github:
  issue: 90
  repo: azettaai/nmn
github_acceptance_import:
  issue: 90
  body_digest: sha256:d7720de322a07475723a18db78d69084d0c8ef7f1aebd798143a3acca82e79f5
  actor: a-root
  imported_at: 2026-08-30T11:56:49Z
---
# Large-magnitude float16 YAT math returns NaNs across backends
## Context
Adopted from GitHub issue #90.

## Summary

Core YAT evaluation overflows intermediate products in float16 in several backends even when the mathematically correct output is finite.

## Reproduction

Using x=[100,100], W=[-100,-99], epsilon=1, with bias/alpha disabled, Torch, Linen, Keras, and TensorFlow return NaN in float16 while float32/NumPy returns approximately 4974.875. NNX already avoids the failure through safer accumulation. Related float16 attention scoring with q=k=100 also produces NaN in affected implementations.

## Impact

Valid large-magnitude low-precision inputs can poison forward outputs and gradients, undermining numerical parity and mixed-precision support.

## Acceptance criteria

- [ ] Accumulate overflow-prone YAT intermediates in a safe compute dtype across affected backends.
- [ ] Preserve genuine NaN inputs rather than masking them.
- [ ] Add float16/bfloat16 forward and gradient parity tests against synchronized float32 references for core/dense and affected attention paths.
- [ ] Verify framework suites and document hardware-specific coverage limits.

## Acceptance
- [x] Accumulate overflow-prone YAT intermediates in a safe compute dtype across affected backends.
- [x] Preserve genuine NaN inputs rather than masking them.
- [x] Add float16/bfloat16 forward and gradient parity tests against synchronized float32 references for core/dense and affected attention paths.
- [x] Verify framework suites and document hardware-specific coverage limits.
## Log
- 2026-08-30T14:03:50Z accepted by a-root
- 2026-08-30T14:03:50Z verified by `KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_low_precision_core_yat.py tests/test_linen/test_low_precision_core_yat.py tests/test_keras/test_low_precision_reductions.py tests/test_nnx/test_attention_regressions.py` (exit 0) in branch codex/acceptance-d2d7 at d2d7e12 — proves that tree builds, not that the work is in trunk
- 2026-08-30T14:03:50Z deliverable: no dacli/044-large-magnitude-float16-yat-math-returns-nans-across-backends branch — nothing to check against master
- 2026-08-30T14:03:50Z completed by a-root
## Verification Evidence
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_low_precision_core_yat.py tests/test_linen/test_low_precision_core_yat.py tests/test_keras/test_low_precision_reductions.py tests/test_nnx/test_attention_regressions.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_low_precision_core_yat.py tests/test_linen/test_low_precision_core_yat.py tests/test_keras/test_low_precision_reductions.py tests/test_nnx/test_attention_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":45565,"artifact_hash":"sha256:385a83f8b21ae62e71dabde72dd520e4cd5cea0dd66f96676fa0c8bb81d39bad","verifier":"a-root","branch":"codex/acceptance-d2d7","commit_sha":"d2d7e125df08b02d6fc13909e05a095a84807484","tree_sha":"7ede1ad4be1f601de3ca374145729a1a08d3446f","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_low_precision_core_yat.py tests/test_linen/test_low_precision_core_yat.py tests/test_keras/test_low_precision_reductions.py tests/test_nnx/test_attention_regressions.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_low_precision_core_yat.py tests/test_linen/test_low_precision_core_yat.py tests/test_keras/test_low_precision_reductions.py tests/test_nnx/test_attention_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":44381,"artifact_hash":"sha256:0b426c96fc5e01a3025c722fba64b11de59d3c03f8f5bf47c11ea648fd5fca9f","verifier":"a-root","branch":"codex/acceptance-d2d7","commit_sha":"d2d7e125df08b02d6fc13909e05a095a84807484","tree_sha":"7ede1ad4be1f601de3ca374145729a1a08d3446f","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"d2d7e125df08b02d6fc13909e05a095a84807484","observed_at":"2026-08-30T14:03:50.141062Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
