---
id: t-01M17A5NFZZD7HZC9F0WJFVVVF
kind: task
created: 2026-08-29T17:48:09Z
created_by: a-root
owner: a-root
github:
  issue: 54
  repo: azettaai/nmn
github_acceptance_import:
  issue: 54
  body_digest: sha256:086520dca8d827761601e1a51c959a4fb74945fea8502894d7682ee1ed666079
  actor: a-root
  imported_at: 2026-08-29T17:48:09Z
estimate: "{optimistic: 3, probable: 5, pessimistic: 8}"
depends_on: "[t-01M17A5HEAF64GWVMC74QK7VKH]"
---
# PyTorch low-precision YAT conv and embedding can return negative or infinite scores
## Context
Adopted from GitHub issue #54.

## Summary

PyTorch YAT convolution and embedding paths compute squared distance as norm sums minus twice the dot product without clamping cancellation errors to zero. Exact input/kernel matches in float16 or bfloat16 can therefore produce negative distances, negative scores, or infinity.

## Affected code

- `src/nmn/torch/layers/_yat_conv_core.py`
- `src/nmn/torch/embed.py`
- Forward and transpose convolution variants that share the core

## Reproduction

On commit `4f0298d` with PyTorch 2.11.0 CPU:

```python
import torch
from nmn.torch import YatConv1D, YatEmbed

for dtype, seed in [(torch.float16, 0), (torch.bfloat16, 9)]:
    torch.manual_seed(seed)
    conv = YatConv1D(
        1, 1, 3, bias=False, use_alpha=False,
        epsilon=1e-5, dtype=dtype,
    )
    conv_out = conv(conv.weight.detach().reshape(1, 1, 3))

    torch.manual_seed(seed)
    embed = YatEmbed(
        1, 3, use_alpha=False, epsilon=1e-5, dtype=dtype,
    )
    embed_out = embed.attend(embed.embedding.detach().reshape(1, 3))
    print(dtype, conv_out.item(), embed_out.item())
```

Observed:

```
torch.float16  -228.125   inf
torch.bfloat16 -2.984375 -97.5
```

For the float16 convolution, the computed distance is `-0.000244140625`; for the bfloat16 embedding it is `-0.015625`.

## Impact

Supported low-precision modes can silently violate the non-negative, finite YAT-score invariant and propagate invalid losses or gradients.

## Acceptance criteria

- [ ] Computed squared distances are clamped to at least zero before epsilon is added.
- [ ] The fix covers convolution, transpose convolution, and spherical/non-spherical embedding paths.
- [ ] CPU float16 and bfloat16 exact-match regression tests produce finite, non-negative outputs and finite gradients.
- [ ] Numerical parity against an fp32 reference is tested away from exact collisions.

## Acceptance
- [x] Computed squared distances are clamped to at least zero before epsilon is added.
- [x] The fix covers convolution, transpose convolution, and spherical/non-spherical embedding paths.
- [x] CPU float16 and bfloat16 exact-match regression tests produce finite, non-negative outputs and finite gradients.
- [x] Numerical parity against an fp32 reference is tested away from exact collisions.
## Log
- 2026-08-29T18:02:28Z dependency edit by a-root (event 01M17AZW0ZZGAC84TZ1434SDAW)
- 2026-08-30T10:47:28Z accepted by a-root
- 2026-08-30T10:47:28Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py` (exit 0) in branch codex/acceptance-dc696 at dc69677 — proves that tree builds, not that the work is in trunk
- 2026-08-30T10:47:28Z deliverable: no dacli/033-pytorch-low-precision-yat-conv-and-embedding-can-return-negative-or-infinite branch — nothing to check against master
- 2026-08-30T10:47:28Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":2291,"artifact_hash":"sha256:f58f381a6e720e248fafffaee74691a8f9c80d73a7cf57617cf3a4ddb4026cd9","verifier":"a-root","branch":"codex/acceptance-dc696","commit_sha":"dc69677f09cc2b37ffc7bced3fddb71265cc061f","tree_sha":"9ba1e98ad5e7fbb58b7b080c3fb54c607919e2cd","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":2080,"artifact_hash":"sha256:bd16be27936932b3f7d1450931ed68e86a6143bfe52af7fb02f68bdace10bc9a","verifier":"a-root","branch":"codex/acceptance-dc696","commit_sha":"dc69677f09cc2b37ffc7bced3fddb71265cc061f","tree_sha":"9ba1e98ad5e7fbb58b7b080c3fb54c607919e2cd","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"dc69677f09cc2b37ffc7bced3fddb71265cc061f","observed_at":"2026-08-30T10:47:28.734301Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
