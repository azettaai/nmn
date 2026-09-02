---
id: t-01M17A5KYY7ZP3BGKHK00MM29F
kind: task
created: 2026-08-29T17:48:08Z
created_by: a-root
owner: a-root
github:
  issue: 62
  repo: azettaai/nmn
github_acceptance_import:
  issue: 62
  body_digest: sha256:a1f85c4480adb06fc61b2d7886379df5d9ad3bf437e72e8543af1dc6af9d919c
  actor: a-root
  imported_at: 2026-08-29T17:48:08Z
estimate: "{optimistic: 3, probable: 5, pessimistic: 8}"
depends_on: "[t-01M17A5PJABDE71M0YXR39TSXA]"
---
# NNX low-precision learnable epsilon initializes to zero across layers
## Context
Adopted from GitHub issue #62.

## Summary

Every NNX module that initializes learnable epsilon in `param_dtype` collapses the default epsilon to zero for float16 and bfloat16.

Affected patterns occur in dense `YatNMN`, `Embed`, `YatConv`, `YatConvTranspose`, `MultiHeadAttention`, and `RotaryYatAttention`.

## Reproduction

For `epsilon=1e-5` and `param_dtype` float16 or bfloat16:

```text
raw epsilon = log(exp(epsilon) - 1) = -inf
softplus(float32(raw epsilon)) = 0
```

Float32 correctly initializes near `-11.511568` and yields approximately `1e-5`.

## Impact

The supposedly strictly positive denominator stabilizer becomes exactly zero. Exact collisions can return infinities and destabilize gradients.

## Acceptance criteria

- [ ] Compute inverse-softplus in stable float32 (for example via `expm1`) before casting/storing.
- [ ] Cover all affected NNX modules.
- [ ] Test float16, bfloat16, and float32 effective epsilon and collision finiteness.

## Acceptance
- [x] Compute inverse-softplus in stable float32 (for example via `expm1`) before casting/storing.
- [x] Cover all affected NNX modules.
- [x] Test float16, bfloat16, and float32 effective epsilon and collision finiteness.
## Log
- 2026-08-29T18:02:29Z dependency edit by a-root (event 01M17AZXDXGPQF843VGYE9VVE0)
- 2026-08-30T11:28:00Z accepted by a-root
- 2026-08-30T11:28:00Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py` (exit 0) in branch codex/acceptance-2e5d at 2e5d913 — proves that tree builds, not that the work is in trunk
- 2026-08-30T11:28:00Z deliverable: no dacli/025-nnx-low-precision-learnable-epsilon-initializes-to-zero-across-layers branch — nothing to check against master
- 2026-08-30T11:28:00Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":27692,"artifact_hash":"sha256:d86b7ec9267278e403a70b5b7244a3664e59481441c07e2a02af33fe8224b3c7","verifier":"a-root","branch":"codex/acceptance-2e5d","commit_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","tree_sha":"9f74c0aa84f8924a738ccd570507d211d879ff29","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":27793,"artifact_hash":"sha256:e857bf0ab932080afeae6664ccd0c056022903da23bec54f38fc317b43f32225","verifier":"a-root","branch":"codex/acceptance-2e5d","commit_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","tree_sha":"9f74c0aa84f8924a738ccd570507d211d879ff29","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","observed_at":"2026-08-30T11:28:00.025537Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
