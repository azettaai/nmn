---
id: t-01M17A5KKBGA9YH1PHN9K06QYJ
kind: task
created: 2026-08-29T17:48:07Z
created_by: a-root
owner: a-root
github:
  issue: 64
  repo: azettaai/nmn
github_acceptance_import:
  issue: 64
  body_digest: sha256:8779ec2221a9316dfa8665ad5c5f752ec2a4ec687e8675b0426df0c170118e74
  actor: a-root
  imported_at: 2026-08-29T17:48:07Z
estimate: "{optimistic: 3, probable: 5, pessimistic: 8}"
depends_on: "[t-01M17A5KQZY2S37QREWQKD16GY]"
---
# NNX grouped YatConv fails for feature_group_count > 1
## Context
Adopted from GitHub issue #64.

## Summary

NNX `YatConv` cannot execute with `feature_group_count > 1` in any spatial dimension.

## Reproduction

For 1D, 2D, and 3D with `in_features=4`, `out_features=4`, and `feature_group_count=2`, the call raises:

```text
rhs output feature dimension size must be a multiple of feature_group_count,
but 1 is not a multiple of 2
```

The dot-product kernel is valid; the patch-norm ones kernel has only one output feature while using grouped convolution.

## Acceptance criteria

- [ ] Produce one patch-norm channel per group and map it to that group's filters.
- [ ] Test 1D/2D/3D grouped forward and gradients against explicit patch references.
- [ ] Cover stride, dilation, and padding.

## Acceptance
- [x] Produce one patch-norm channel per group and map it to that group's filters.
- [x] Test 1D/2D/3D grouped forward and gradients against explicit patch references.
- [x] Cover stride, dilation, and padding.
## Log
- 2026-08-29T18:02:30Z dependency edit by a-root (event 01M17AZXS9W8W04KD4RWKPYZJ5)
- 2026-08-30T11:27:04Z accepted by a-root
- 2026-08-30T11:27:04Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py` (exit 0) in branch codex/acceptance-2e5d at 2e5d913 — proves that tree builds, not that the work is in trunk
- 2026-08-30T11:27:04Z deliverable: no dacli/023-nnx-grouped-yatconv-fails-for-feature-group-count-1 branch — nothing to check against master
- 2026-08-30T11:27:04Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":25801,"artifact_hash":"sha256:b07cf115b174bd9b95496f212f94ef18ab7c9ae70510eacb5ad2cdf09e424072","verifier":"a-root","branch":"codex/acceptance-2e5d","commit_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","tree_sha":"9f74c0aa84f8924a738ccd570507d211d879ff29","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_nnx_layer_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":26117,"artifact_hash":"sha256:43404f2c02a3b2d94000062b7028e18cd1c9f0edaeb0a2bc39e10563aef53751","verifier":"a-root","branch":"codex/acceptance-2e5d","commit_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","tree_sha":"9f74c0aa84f8924a738ccd570507d211d879ff29","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","observed_at":"2026-08-30T11:27:04.519308Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
