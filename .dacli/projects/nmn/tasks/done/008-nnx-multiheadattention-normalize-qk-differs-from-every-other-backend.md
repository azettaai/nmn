---
id: t-01M17A5FYAQP1ZD1ZJRTGZE3R1
kind: task
created: 2026-08-29T17:48:03Z
created_by: a-root
owner: a-root
github:
  issue: 79
  repo: azettaai/nmn
github_acceptance_import:
  issue: 79
  body_digest: sha256:b8da8a53124673e1e3cd219ecb40ec71ecf009607b2b6320cb6be966e4c7115e
  actor: a-root
  imported_at: 2026-08-29T17:48:03Z
estimate: "{optimistic: 2, probable: 4, pessimistic: 7}"
depends_on: "[t-01M17A5JB6HSKPFK3C0PABK89V]"
---
# NNX MultiHeadAttention normalize_qk differs from every other backend
## Context
Adopted from GitHub issue #79.

## Summary

`normalize_qk=True` has different semantics in NNX `MultiHeadAttention` than in every other backend.

- PyTorch, Linen, Keras, TensorFlow, and MLX perform per-head L2 normalization.
- NNX creates learnable `LayerNorm` modules for Q and K.
- NNX's own standalone `normalize_qk` function performs L2 normalization.

## Impact

The same public option changes values, parameters, optimizer state, and numerical behavior across supposedly equivalent implementations.

## Acceptance criteria

- [ ] Align the option across backends, or rename/document distinct LayerNorm versus L2 modes.
- [ ] Add cross-backend parity tests with synchronized projection weights.

## Acceptance
- [x] Align the option across backends, or rename/document distinct LayerNorm versus L2 modes.
- [x] Add cross-backend parity tests with synchronized projection weights.
## Log
- 2026-08-29T18:02:31Z dependency edit by a-root (event 01M17AZYT6530FYGGQZ9EVZCQ6)
- 2026-08-30T09:11:16Z accepted by a-root
- 2026-08-30T09:11:16Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'normalize_qk'` (exit 0) in branch codex/acceptance-b88 at b88e5ed — proves that tree builds, not that the work is in trunk
- 2026-08-30T09:11:16Z deliverable: no dacli/008-nnx-multiheadattention-normalize-qk-differs-from-every-other-backend branch — nothing to check against master
- 2026-08-30T09:11:16Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'normalize_qk'","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'normalize_qk'"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":5495,"artifact_hash":"sha256:54a1aeb90a3904fd28af0cd4e45baa71dc31df6828f09956e951c32d325e1182","verifier":"a-root","branch":"codex/acceptance-b88","commit_sha":"b88e5eda007cb2baa1205d9e803d4aff81665aad","tree_sha":"fb080c6983893af1e0e3e4bc5eacb87a42d2d970","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'normalize_qk'","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'normalize_qk'"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":5838,"artifact_hash":"sha256:f0392ddd11b005c3bc67cd28eafbf109f3aa3e553d39632a3ea3d106e26fb596","verifier":"a-root","branch":"codex/acceptance-b88","commit_sha":"b88e5eda007cb2baa1205d9e803d4aff81665aad","tree_sha":"fb080c6983893af1e0e3e4bc5eacb87a42d2d970","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"b88e5eda007cb2baa1205d9e803d4aff81665aad","observed_at":"2026-08-30T09:11:16.321713Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
