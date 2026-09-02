---
id: t-01M17A5KQZY2S37QREWQKD16GY
kind: task
created: 2026-08-29T17:48:07Z
created_by: a-root
owner: a-root
github:
  issue: 63
  repo: azettaai/nmn
github_acceptance_import:
  issue: 63
  body_digest: sha256:8d223c3ac2ca0c11e060f0dfeeb0736e44c48094ebd03be8a25de480d86577a3
  actor: a-root
  imported_at: 2026-08-29T17:48:07Z
estimate: "{optimistic: 3, probable: 5, pessimistic: 8}"
depends_on: "[t-01M17A5KYY7ZP3BGKHK00MM29F]"
---
# NNX conv and embedding distances can become negative in low precision
## Context
Adopted from GitHub issue #63.

## Summary

NNX convolution, transpose-convolution, and embedding-attend paths reconstruct squared Euclidean distance as norm sums minus twice the dot product without clamping cancellation errors to zero.

## Reproduction

- `YatConv` exact input/kernel: float16 seed 0 returns `inf`; bfloat16 seed 8 returns `-69`.
- `Embed.attend` exact query/embedding: float16 seed 0 returns `inf`; bfloat16 seed 8 returns `-1872`.
- `Embed(spherical=True)` also divides zero vectors by an unguarded norm and returns NaNs.

The same unclamped distance expression is present in `YatConvTranspose`.

## Impact

Supported low-precision dtypes can violate the finite, non-negative YAT-score invariant. Spherical zero/padding vectors can poison the whole output.

## Acceptance criteria

- [ ] Clamp reconstructed distances before adding epsilon in all three families.
- [ ] Guard spherical embedding normalization against zero norms.
- [ ] Add fp16/bf16 exact-collision forward and gradient tests plus fp32 parity tests.

## Acceptance
- [x] Clamp reconstructed distances before adding epsilon in all three families.
- [x] Guard spherical embedding normalization against zero norms.
- [x] Add fp16/bf16 exact-collision forward and gradient tests plus fp32 parity tests.
## Log
- 2026-08-29T18:02:29Z dependency edit by a-root (event 01M17AZXM78XWCD2Z2EW6W2NGM)
- 2026-08-30T11:27:31Z accepted by a-root
- 2026-08-30T11:27:31Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py` (exit 0) in branch codex/acceptance-2e5d at 2e5d913 — proves that tree builds, not that the work is in trunk
- 2026-08-30T11:27:31Z deliverable: no dacli/024-nnx-conv-and-embedding-distances-can-become-negative-in-low-precision branch — nothing to check against master
- 2026-08-30T11:27:31Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":27730,"artifact_hash":"sha256:3f7b3f003363d02ab3320e93f10d6b46c820368d96e93a8be220236b917c5394","verifier":"a-root","branch":"codex/acceptance-2e5d","commit_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","tree_sha":"9f74c0aa84f8924a738ccd570507d211d879ff29","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":27105,"artifact_hash":"sha256:4c0460ef9a587430e6da2f2d35e33552bef48575c2bc3f03ea0b37ed57df0def","verifier":"a-root","branch":"codex/acceptance-2e5d","commit_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","tree_sha":"9f74c0aa84f8924a738ccd570507d211d879ff29","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","observed_at":"2026-08-30T11:27:31.929404Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
