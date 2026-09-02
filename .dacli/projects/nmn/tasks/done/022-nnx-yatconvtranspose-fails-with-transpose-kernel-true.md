---
id: t-01M17A5KEV72P6F3Z6FMJCWCS4
kind: task
created: 2026-08-29T17:48:07Z
created_by: a-root
owner: a-root
github:
  issue: 65
  repo: azettaai/nmn
github_acceptance_import:
  issue: 65
  body_digest: sha256:9d2ef939289c12f74c2d53a75786e36fa242acdab11e961b5f9e64c31b3be684
  actor: a-root
  imported_at: 2026-08-29T17:48:07Z
estimate: "{optimistic: 3, probable: 5, pessimistic: 8}"
depends_on: "[t-01M17A5KKBGA9YH1PHN9K06QYJ]"
---
# NNX YatConvTranspose fails with transpose_kernel=True
## Context
Adopted from GitHub issue #65.

## Summary

NNX `YatConvTranspose(..., transpose_kernel=True)` fails for 1D, 2D, and 3D.

## Reproduction

With `in_features=2`, `out_features=3`, all three dimensional variants raise a `conv_general_dilated` input-feature mismatch (`2 // 1 != 1`). The primary transpose convolution accepts the transposed layout, but the patch-squared-sum ones kernel puts the singleton on the wrong logical axis after kernel transposition.

## Acceptance criteria

- [ ] Construct the patch-norm kernel in the correct layout for both values of `transpose_kernel`.
- [ ] Compare forwards and gradients for 1D/2D/3D against the non-transposed equivalent.

## Acceptance
- [x] Construct the patch-norm kernel in the correct layout for both values of `transpose_kernel`.
- [x] Compare forwards and gradients for 1D/2D/3D against the non-transposed equivalent.
## Log
- 2026-08-29T18:02:30Z dependency edit by a-root (event 01M17AZXYEPJWJ1EYFV08ZA1BX)
- 2026-08-30T11:26:38Z accepted by a-root
- 2026-08-30T11:26:38Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py` (exit 0) in branch codex/acceptance-2e5d at 2e5d913 — proves that tree builds, not that the work is in trunk
- 2026-08-30T11:26:38Z deliverable: no dacli/022-nnx-yatconvtranspose-fails-with-transpose-kernel-true branch — nothing to check against master
- 2026-08-30T11:26:38Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":25211,"artifact_hash":"sha256:7e7d31b6fc72686716bc4c12d8ea761ff8cbfeea23234beb11b5b3b2543cb8d6","verifier":"a-root","branch":"codex/acceptance-2e5d","commit_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","tree_sha":"9f74c0aa84f8924a738ccd570507d211d879ff29","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":26689,"artifact_hash":"sha256:3f4f0a0b4df195e27e885cfb7ebfb7f38541f241738fa6253cb3f735010a137f","verifier":"a-root","branch":"codex/acceptance-2e5d","commit_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","tree_sha":"9f74c0aa84f8924a738ccd570507d211d879ff29","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","observed_at":"2026-08-30T11:26:38.102971Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
