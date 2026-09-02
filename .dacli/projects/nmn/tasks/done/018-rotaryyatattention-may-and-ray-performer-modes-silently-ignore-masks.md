---
id: t-01M17A5JB6HSKPFK3C0PABK89V
kind: task
created: 2026-08-29T17:48:06Z
created_by: a-root
owner: a-root
github:
  issue: 69
  repo: azettaai/nmn
github_acceptance_import:
  issue: 69
  body_digest: sha256:3a79c7946797e2d97e4fdba743e2fe8d3742158081d481a799c33e7b93deaedb
  actor: a-root
  imported_at: 2026-08-29T17:48:06Z
estimate: "{optimistic: 3, probable: 5, pessimistic: 8}"
depends_on: "[t-01M17A5K252ZQH96AWPBX5S4KC]"
---
# RotaryYatAttention MAY and RAY performer modes silently ignore masks
## Context
Adopted from GitHub issue #69.

## Summary

`RotaryYatAttention` silently ignores its public `mask` argument in Maclaurin (MAY) and radial (RAY) performer modes.

## Reproduction

For both `performer_kind="maclaurin"` and `"radial"`, an all-True mask and an identity-only mask produce bit-identical outputs (`max diff = 0`). These branches call linear attention with `causal=self.causal` only; SLAY and quadratic branches pass the mask.

## Impact

Padding, packed-sequence, and access-control masks are silently bypassed.

## Acceptance criteria

- [ ] Correctly support compatible masks, or reject unsupported masks explicitly.
- [ ] Add mask-sensitivity tests for every performer kind, including decode and padding masks.

## Acceptance
- [x] Correctly support compatible masks, or reject unsupported masks explicitly.
- [x] Add mask-sensitivity tests for every performer kind, including decode and padding masks.
## Log
- 2026-08-29T18:02:30Z dependency edit by a-root (event 01M17AZYMJ5JW3NK1SP7KBWF14)
- 2026-08-30T09:11:23Z accepted by a-root
- 2026-08-30T09:11:23Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'performer'` (exit 0) in branch codex/acceptance-b88 at b88e5ed — proves that tree builds, not that the work is in trunk
- 2026-08-30T09:11:23Z deliverable: no dacli/018-rotaryyatattention-may-and-ray-performer-modes-silently-ignore-masks branch — nothing to check against master
- 2026-08-30T09:11:23Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'performer'","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'performer'"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":7344,"artifact_hash":"sha256:9f180d3b1b4200018a25c76093cf32ad0b0962b041b87ec053c120da77c1d33d","verifier":"a-root","branch":"codex/acceptance-b88","commit_sha":"b88e5eda007cb2baa1205d9e803d4aff81665aad","tree_sha":"fb080c6983893af1e0e3e4bc5eacb87a42d2d970","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'performer'","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'performer'"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":7115,"artifact_hash":"sha256:bf2b40460066fdaa054c19a5793b34d75d20b5f1608c0320dadbeeee5037c8bf","verifier":"a-root","branch":"codex/acceptance-b88","commit_sha":"b88e5eda007cb2baa1205d9e803d4aff81665aad","tree_sha":"fb080c6983893af1e0e3e4bc5eacb87a42d2d970","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"b88e5eda007cb2baa1205d9e803d4aff81665aad","observed_at":"2026-08-30T09:11:23.840759Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
