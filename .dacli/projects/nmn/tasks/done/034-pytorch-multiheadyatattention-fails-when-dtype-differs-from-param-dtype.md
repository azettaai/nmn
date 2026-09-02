---
id: t-01M17A5NMCRB8722Q6T0A6W6CP
kind: task
created: 2026-08-29T17:48:09Z
created_by: a-root
owner: a-root
github:
  issue: 53
  repo: azettaai/nmn
github_acceptance_import:
  issue: 53
  body_digest: sha256:32566491da2cb2f52583534aee7b196683295f931cdfad838009130636b5c5d3
  actor: a-root
  imported_at: 2026-08-29T17:48:09Z
estimate: "{optimistic: 3, probable: 5, pessimistic: 8}"
depends_on: "[t-01M17A5NFZZD7HZC9F0WJFVVVF]"
---
# PyTorch MultiHeadYatAttention fails when dtype differs from param_dtype
## Context
Adopted from GitHub issue #53.

## Summary

`MultiHeadYatAttention` accepts separate `dtype` and `param_dtype` arguments, but its projection layers receive raw input before any compute-dtype promotion. When parameter storage dtype differs from input/compute dtype, PyTorch fails in the first linear projection.

## Affected code

- `src/nmn/torch/attention/multi_head.py`

## Reproduction

On commit `4f0298d` with PyTorch 2.11.0:

```python
import torch
from nmn.torch import MultiHeadYatAttention

layer = MultiHeadYatAttention(
    4, 2,
    dtype=torch.float32,
    param_dtype=torch.float64,
)
layer(torch.randn(1, 2, 4, dtype=torch.float32))
```

Observed:

```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Double
```

The Q/K/V results are promoted only after the projections, so that promotion cannot prevent the failure.

## Impact

The advertised compute/storage dtype separation is unusable for PyTorch attention and blocks mixed-precision configurations that work in other layer families.

## Acceptance criteria

- [ ] Unequal `dtype` and `param_dtype` complete forward and backward without dtype errors.
- [ ] Outputs use the requested compute dtype while parameters retain storage dtype.
- [ ] Q/K/V and output projection math follows a documented promotion policy.
- [ ] State-dict round trips preserve both dtype behavior and numerical output.

## Acceptance
- [x] Unequal `dtype` and `param_dtype` complete forward and backward without dtype errors.
- [x] Outputs use the requested compute dtype while parameters retain storage dtype.
- [x] Q/K/V and output projection math follows a documented promotion policy.
- [x] State-dict round trips preserve both dtype behavior and numerical output.
## Log
- 2026-08-29T18:02:28Z dependency edit by a-root (event 01M17AZW70ZKRAMGMN5SFBFXXE)
- 2026-08-30T10:47:31Z accepted by a-root
- 2026-08-30T10:47:31Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py` (exit 0) in branch codex/acceptance-dc696 at dc69677 — proves that tree builds, not that the work is in trunk
- 2026-08-30T10:47:31Z deliverable: no dacli/034-pytorch-multiheadyatattention-fails-when-dtype-differs-from-param-dtype branch — nothing to check against master
- 2026-08-30T10:47:31Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":2342,"artifact_hash":"sha256:e2fbbf32c21701a57baf904af98b5ff88d021412786e3cfcd38e8f7e70b8729f","verifier":"a-root","branch":"codex/acceptance-dc696","commit_sha":"dc69677f09cc2b37ffc7bced3fddb71265cc061f","tree_sha":"9ba1e98ad5e7fbb58b7b080c3fb54c607919e2cd","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":2205,"artifact_hash":"sha256:f23fb2aab773b5d4614f23d4bbf43357d5095fc2d2d78920afbd8ee99f4e0a87","verifier":"a-root","branch":"codex/acceptance-dc696","commit_sha":"dc69677f09cc2b37ffc7bced3fddb71265cc061f","tree_sha":"9ba1e98ad5e7fbb58b7b080c3fb54c607919e2cd","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"dc69677f09cc2b37ffc7bced3fddb71265cc061f","observed_at":"2026-08-30T10:47:31.307955Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
