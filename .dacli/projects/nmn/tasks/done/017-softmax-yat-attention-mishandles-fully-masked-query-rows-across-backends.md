---
id: t-01M17A5HXEWFQJKHX8AK8QZPCH
kind: task
created: 2026-08-29T17:48:05Z
created_by: a-root
owner: a-root
github:
  issue: 70
  repo: azettaai/nmn
github_acceptance_import:
  issue: 70
  body_digest: sha256:46a171f852fb194f6aa7df62747623db2c171b12a8667c28c29b7f45cf1e4c15
  actor: a-root
  imported_at: 2026-08-29T17:48:05Z
estimate: "{optimistic: 5, probable: 8, pessimistic: 13}"
depends_on: "[t-01M17A5FYAQP1ZD1ZJRTGZE3R1, t-01M17A5NMCRB8722Q6T0A6W6CP, t-01M17A5N54KX8WK3NGECV5R935, t-01M17A5NRQ0VEVMA1RQ229DHE5, t-01M17A5MRHNCWVQZM9KBTCBCFD, t-01M17A5GAKHM36J49R8M01N79V]"
---
# Softmax YAT attention mishandles fully masked query rows across backends
## Context
Adopted from GitHub issue #70.

## Summary

Softmax YAT attention mishandles query rows where every key is masked.

## Reproduction

For one query, two keys, and an all-False boolean mask:

- PyTorch returns `[NaN, NaN]` after softmax of all `-inf`.
- Keras/JAX and NNX return `[0.5, 0.5]` because equal finite negative sentinels normalize to a uniform distribution.
- Linen delegates to NNX; TensorFlow and MLX use the same finite-sentinel pattern in source.
- NNX L1/softermax already return zeros in this case.

## Impact

Fully padded or intentionally disabled query rows either poison outputs or leak the mean of masked values.

## Acceptance criteria

- [ ] Define and implement a cross-backend zero-output policy for fully masked rows.
- [ ] Ensure weights are zero and gradients finite.
- [ ] Test functional and module APIs for self/cross attention and every supported normalization.

## Acceptance
- [x] Define and implement a cross-backend zero-output policy for fully masked rows.
- [x] Ensure weights are zero and gradients finite.
- [x] Test functional and module APIs for self/cross attention and every supported normalization.
## Log
- 2026-08-29T18:02:32Z dependency edit by a-root (event 01M17B004G8ZBD73RD8ZY1RARJ)
- 2026-08-29T18:02:32Z dependency edit by a-root (event 01M17B00A02JTQWMNNP3Q1XWZ0)
- 2026-08-29T18:02:32Z dependency edit by a-root (event 01M17B00F8RJ42359XSYFYF93P)
- 2026-08-29T18:02:33Z dependency edit by a-root (event 01M17B00ND18SJDSPNT8M9EQRQ)
- 2026-08-29T18:02:33Z dependency edit by a-root (event 01M17B00V2TNGF0RCX84FE1W8K)
- 2026-08-29T18:02:33Z dependency edit by a-root (event 01M17B0104E1RX9FSG7M1P677E)
- 2026-08-30T14:49:35Z accepted by a-root
- 2026-08-30T14:49:35Z verified by `env PYTHONPATH=src python3 -m pytest -q tests/test_nnx/test_fully_masked_attention.py tests/test_torch/test_attention.py tests/test_linen/test_attention.py` (exit 0) in branch codex/acceptance-d6278 at d6278a2 — proves that tree builds, not that the work is in trunk
- 2026-08-30T14:49:35Z deliverable: no dacli/017-softmax-yat-attention-mishandles-fully-masked-query-rows-across-backends branch — nothing to check against master
- 2026-08-30T14:49:35Z completed by a-root
## Verification Evidence
{"command":"env PYTHONPATH=src python3 -m pytest -q tests/test_nnx/test_fully_masked_attention.py tests/test_torch/test_attention.py tests/test_linen/test_attention.py","argv":["sh","-c","env PYTHONPATH=src python3 -m pytest -q tests/test_nnx/test_fully_masked_attention.py tests/test_torch/test_attention.py tests/test_linen/test_attention.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":20710,"artifact_hash":"sha256:ef3fde4b3a967750a95cf88a103b832ee5e5372a9298f70206b987769da45922","verifier":"a-root","branch":"codex/acceptance-d6278","commit_sha":"d6278a2f8aa38a736e923f31b739ae6bd4b39a58","tree_sha":"96426f20f1935341a5929904239e705341dc96b0","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"env PYTHONPATH=src python3 -m pytest -q tests/test_nnx/test_fully_masked_attention.py tests/test_torch/test_attention.py tests/test_linen/test_attention.py","argv":["sh","-c","env PYTHONPATH=src python3 -m pytest -q tests/test_nnx/test_fully_masked_attention.py tests/test_torch/test_attention.py tests/test_linen/test_attention.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":19384,"artifact_hash":"sha256:15423583030645ca8be2b2d8501ab91da1da3206093b980394c965d5ed2b6378","verifier":"a-root","branch":"codex/acceptance-d6278","commit_sha":"d6278a2f8aa38a736e923f31b739ae6bd4b39a58","tree_sha":"96426f20f1935341a5929904239e705341dc96b0","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"d6278a2f8aa38a736e923f31b739ae6bd4b39a58","observed_at":"2026-08-30T14:49:35.958643Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
