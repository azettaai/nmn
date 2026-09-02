---
id: t-01M17A5K252ZQH96AWPBX5S4KC
kind: task
created: 2026-08-29T17:48:07Z
created_by: a-root
owner: a-root
github:
  issue: 67
  repo: azettaai/nmn
github_acceptance_import:
  issue: 67
  body_digest: sha256:035c0e68f0cfda4da4ad12b46ba86d2a3b31ca9561ba28c33d4c2b5816972bac
  actor: a-root
  imported_at: 2026-08-29T17:48:07Z
estimate: "{optimistic: 5, probable: 8, pessimistic: 13}"
depends_on: "[t-01M17A5JQ0AR0H1ZSGEZ841P62]"
---
# NNX autoregressive attention caches fail and rotary decode uses wrong RoPE positions
## Context
Adopted from GitHub issue #67.

## Summary

NNX autoregressive attention is blocked at cache initialization, and rotary decoding remains numerically wrong after bypassing that blocker.

## Reproduction

On Flax 0.12/JAX 0.9.2, both `MultiHeadAttention.init_cache(...)` and `RotaryYatAttention.init_cache(...)` raise because `cached_key/value/index` were created as static `None` attributes and are later replaced with `nnx.Cache` data variables.

After locally initializing those attributes with `nnx.data`, length-4 full causal rotary attention and token-by-token decode agree only at token 0; subsequent per-step max errors are about `0.99`, `0.78`, and `1.51`.

Decode sets `position_offset=0`, so every one-token query is rotated at position zero instead of `cur_index`.

## Acceptance criteria

- [ ] Initialize cache attributes with stable NNX data-variable status.
- [ ] Use the current cache index for the decoded query's RoPE position while cached keys retain positions `0..index`.
- [ ] Add full-causal versus incremental parity, overflow, batch, and JIT tests for both modules.

## Acceptance
- [x] Initialize cache attributes with stable NNX data-variable status.
- [x] Use the current cache index for the decoded query's RoPE position while cached keys retain positions `0..index`.
- [x] Add full-causal versus incremental parity, overflow, batch, and JIT tests for both modules.
## Log
- 2026-08-29T18:02:30Z dependency edit by a-root (event 01M17AZYEEGME10DNJQ8D1A1H4)
- 2026-08-30T09:11:39Z accepted by a-root
- 2026-08-30T09:11:39Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'decode or causal_matches'` (exit 0) in branch codex/acceptance-b88 at b88e5ed — proves that tree builds, not that the work is in trunk
- 2026-08-30T09:11:39Z deliverable: no dacli/020-nnx-autoregressive-attention-caches-fail-and-rotary-decode-uses-wrong-rope branch — nothing to check against master
- 2026-08-30T09:11:39Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'decode or causal_matches'","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'decode or causal_matches'"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":12122,"artifact_hash":"sha256:8dcd93f0c0bdce99e403de896f06acb9b8c2325802fd7cec922b3e61fc475af6","verifier":"a-root","branch":"codex/acceptance-b88","commit_sha":"b88e5eda007cb2baa1205d9e803d4aff81665aad","tree_sha":"fb080c6983893af1e0e3e4bc5eacb87a42d2d970","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'decode or causal_matches'","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'decode or causal_matches'"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":11578,"artifact_hash":"sha256:da133ada489012d2f2437d32b7a1c9833143a669f4398a8f512911cb276f73d5","verifier":"a-root","branch":"codex/acceptance-b88","commit_sha":"b88e5eda007cb2baa1205d9e803d4aff81665aad","tree_sha":"fb080c6983893af1e0e3e4bc5eacb87a42d2d970","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"b88e5eda007cb2baa1205d9e803d4aff81665aad","observed_at":"2026-08-30T09:11:39.643195Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
