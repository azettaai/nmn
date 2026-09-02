---
id: t-01M17A5M91HYTJGV2NF3WGDMD7
kind: task
created: 2026-08-29T17:48:08Z
created_by: a-root
owner: a-root
github:
  issue: 61
  repo: azettaai/nmn
github_acceptance_import:
  issue: 61
  body_digest: sha256:63a7a394525e7ef022f1e7495d0866a801cc5ca38a6cc2e57eaeb1f15d041672
  actor: a-root
  imported_at: 2026-08-29T17:48:08Z
estimate: "{optimistic: 5, probable: 8, pessimistic: 13}"
depends_on: "[t-01M17A5FYAQP1ZD1ZJRTGZE3R1]"
---
# NNX DropConnect is deterministic or a no-op across layer families
## Context
Adopted from GitHub issue #61.

## Summary

NNX DropConnect is not stochastic across the advertised layer families.

- `YatNMN`, `YatConv`, and `YatConvTranspose` store one `rngs.params()` key at construction and reuse it for every training call.
- `MultiHeadAttention` stores `use_dropconnect` / `dropconnect_rate` but never masks any projection kernel.

## Reproduction

With `drop_rate=0.5`, two non-deterministic dense calls are bit-identical (`maxdiff=0`). Two models with the same params seed but different dropout seeds also store equal DropConnect keys and produce equal training outputs. `MultiHeadAttention` likewise produces bit-identical outputs with different dropout RNG streams.

## Impact

Training silently uses a fixed pruning mask, or no mask at all for attention, instead of DropConnect regularization.

## Acceptance criteria

- [ ] Consume a mutable dropout stream on every non-deterministic call.
- [ ] Do not derive DropConnect masks from the params stream.
- [ ] Implement or reject the attention option instead of silently ignoring it.
- [ ] Add stochasticity, deterministic-mode, JIT, and independent-stream tests.

## Acceptance
- [x] Consume a mutable dropout stream on every non-deterministic call.
- [x] Do not derive DropConnect masks from the params stream.
- [x] Implement or reject the attention option instead of silently ignoring it.
- [x] Add stochasticity, deterministic-mode, JIT, and independent-stream tests.
## Log
- 2026-08-29T18:02:31Z dependency edit by a-root (event 01M17AZZ0W6YBB4MDSRYZ2WPQZ)
- 2026-08-30T11:28:29Z accepted by a-root
- 2026-08-30T11:28:29Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py` (exit 0) in branch codex/acceptance-2e5d at 2e5d913 — proves that tree builds, not that the work is in trunk
- 2026-08-30T11:28:29Z deliverable: no dacli/026-nnx-dropconnect-is-deterministic-or-a-no-op-across-layer-families branch — nothing to check against master
- 2026-08-30T11:28:29Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":26762,"artifact_hash":"sha256:798ac80362e4b97e0be8f3e3902dd10de4e19dd9a1c753151f180c160bcb4e4d","verifier":"a-root","branch":"codex/acceptance-2e5d","commit_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","tree_sha":"9f74c0aa84f8924a738ccd570507d211d879ff29","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":29281,"artifact_hash":"sha256:e3821be0bf7ebffbe28418688c79cc6330e816117af8bf2a29a768f5b3339d4e","verifier":"a-root","branch":"codex/acceptance-2e5d","commit_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","tree_sha":"9f74c0aa84f8924a738ccd570507d211d879ff29","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","observed_at":"2026-08-30T11:28:29.600586Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
