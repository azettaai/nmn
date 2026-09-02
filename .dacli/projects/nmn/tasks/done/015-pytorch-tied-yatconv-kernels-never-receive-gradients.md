---
id: t-01M17A5HEAF64GWVMC74QK7VKH
kind: task
created: 2026-08-29T17:48:05Z
created_by: a-root
owner: a-root
github:
  issue: 72
  repo: azettaai/nmn
github_acceptance_import:
  issue: 72
  body_digest: sha256:bd72e3ef53203d2e8799951e3d93b4110380b3781dac4cf778ebba9a59498aed
  actor: a-root
  imported_at: 2026-08-29T17:48:05Z
estimate: "{optimistic: 3, probable: 5, pessimistic: 8}"
depends_on: "[t-01M17A5HPD5E247R54ZS01JPCA]"
---
# PyTorch tied YatConv kernels never receive gradients
## Context
Adopted from GitHub issue #72.

## Summary

PyTorch tied `YatConv1D/2D/3D` kernels never receive gradients.

## Reproduction

After `sum(layer(x)).backward()` with `tie_kernel_bank=True`, `layer.weight.grad is None` for all dimensions while bias/alpha gradients exist. `forward()` temporarily replaces the shared weight with `nn.Parameter(original_weight[slice])`; autograd accumulates into that temporary leaf, then the `finally` block discards it.

An explicit `kernel_bank_size` larger than `out_channels` also fails with default bias because the parent convolution allocates bank-sized bias while the sliced dot output has only layer-sized channels.

## Acceptance criteria

- [ ] Slice without creating a detached leaf so gradients accumulate into the shared bank.
- [ ] Allocate/slice bias according to the layer's actual output channels.
- [ ] Test shared gradient accumulation, optimizers, different consumer widths, and all dimensions.

## Acceptance
- [x] Slice without creating a detached leaf so gradients accumulate into the shared bank.
- [x] Allocate/slice bias according to the layer's actual output channels.
- [x] Test shared gradient accumulation, optimizers, different consumer widths, and all dimensions.
## Log
- 2026-08-29T18:02:28Z dependency edit by a-root (event 01M17AZVV6G13HAN6DVC9YGRZS)
- 2026-08-30T10:47:23Z accepted by a-root
- 2026-08-30T10:47:23Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py` (exit 0) in branch codex/acceptance-dc696 at dc69677 — proves that tree builds, not that the work is in trunk
- 2026-08-30T10:47:23Z deliverable: no dacli/015-pytorch-tied-yatconv-kernels-never-receive-gradients branch — nothing to check against master
- 2026-08-30T10:47:23Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":2347,"artifact_hash":"sha256:dcba4455370810f1c169170f87b44c2784dee749f9223d61f4f2a6509076e066","verifier":"a-root","branch":"codex/acceptance-dc696","commit_sha":"dc69677f09cc2b37ffc7bced3fddb71265cc061f","tree_sha":"9ba1e98ad5e7fbb58b7b080c3fb54c607919e2cd","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":2043,"artifact_hash":"sha256:78291bbc30401ca2e7590bed3a134f88778b11c9924497a3efb27457cb4d05ec","verifier":"a-root","branch":"codex/acceptance-dc696","commit_sha":"dc69677f09cc2b37ffc7bced3fddb71265cc061f","tree_sha":"9ba1e98ad5e7fbb58b7b080c3fb54c607919e2cd","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"dc69677f09cc2b37ffc7bced3fddb71265cc061f","observed_at":"2026-08-30T10:47:23.817421Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
