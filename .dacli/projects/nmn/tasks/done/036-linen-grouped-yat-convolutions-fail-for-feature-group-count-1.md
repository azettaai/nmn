---
id: t-01M17A5NX8ER8S1W4KK5M51EPH
kind: task
created: 2026-08-29T17:48:10Z
created_by: a-root
owner: a-root
github:
  issue: 51
  repo: azettaai/nmn
github_acceptance_import:
  issue: 51
  body_digest: sha256:bb1cf8f3cb8a729f550f00bf8451a1919c33d15b8d9cbdd17ddc5c195e7dc62d
  actor: a-root
  imported_at: 2026-08-29T17:48:10Z
estimate: "{optimistic: 3, probable: 5, pessimistic: 8}"
depends_on: "[t-01M17A5NRQ0VEVMA1RQ229DHE5]"
---
# Linen grouped YAT convolutions fail for feature_group_count > 1
## Context
Adopted from GitHub issue #51.

## Summary

All three Linen forward YAT convolution layers fail when `feature_group_count > 1`. The helper convolution used to compute each input patch norm creates an `ones_kernel` with one output feature, but JAX requires the RHS output feature count to be a multiple of `feature_group_count`.

## Affected code

- `src/nmn/linen/conv.py`
- `YatConv1D`, `YatConv2D`, and `YatConv3D`

## Reproduction

On commit `4f0298d`:

```python
import jax
import jax.numpy as jnp
from nmn.linen import YatConv1D, YatConv2D, YatConv3D

cases = [
    (YatConv1D, jnp.ones((1, 8, 4)), (3,)),
    (YatConv2D, jnp.ones((1, 8, 8, 4)), (3, 3)),
    (YatConv3D, jnp.ones((1, 6, 6, 6, 4)), (3, 3, 3)),
]
for cls, x, kernel_size in cases:
    layer = cls(
        features=4,
        kernel_size=kernel_size,
        feature_group_count=2,
        padding="SAME",
    )
    variables = layer.init(jax.random.key(0), x)
    print(layer.apply(variables, x).shape)
```

Each layer raises:

```
ValueError: conv_general_dilated rhs output feature dimension size must be a
multiple of feature_group_count, but 1 is not a multiple of 2.
```

The main convolution kernel is valid; the failure comes from the patch-norm `ones_kernel` whose last dimension is hard-coded to 1.

## Impact

Grouped and depthwise-style Linen YAT convolutions cannot run despite `feature_group_count` being a public option.

## Acceptance criteria

- [ ] Patch norms are computed independently for every input group without mixing channels.
- [ ] `YatConv1D`, `YatConv2D`, and `YatConv3D` run with `feature_group_count > 1`.
- [ ] Numerical parity tests compare grouped outputs and gradients against an explicit patch-based YAT reference.
- [ ] Invalid input/output channel divisibility is rejected with a clear error.

## Acceptance
- [x] Patch norms are computed independently for every input group without mixing channels.
- [x] `YatConv1D`, `YatConv2D`, and `YatConv3D` run with `feature_group_count > 1`.
- [x] Numerical parity tests compare grouped outputs and gradients against an explicit patch-based YAT reference.
- [x] Invalid input/output channel divisibility is rejected with a clear error.
## Log
- 2026-08-29T18:02:31Z dependency edit by a-root (event 01M17AZZ6HS9WK5FEVKSQS7CX0)
- 2026-08-30T11:26:11Z accepted by a-root
- 2026-08-30T11:26:11Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_linen/test_regression_issues_51_52.py` (exit 0) in branch codex/acceptance-2e5d at 2e5d913 — proves that tree builds, not that the work is in trunk
- 2026-08-30T11:26:11Z deliverable: no dacli/036-linen-grouped-yat-convolutions-fail-for-feature-group-count-1 branch — nothing to check against master
- 2026-08-30T11:26:11Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_linen/test_regression_issues_51_52.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_linen/test_regression_issues_51_52.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":12914,"artifact_hash":"sha256:dfd06b24da8f252793b99b28f7f40824a3b78a476af97afdcf0ad864141a60c4","verifier":"a-root","branch":"codex/acceptance-2e5d","commit_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","tree_sha":"9f74c0aa84f8924a738ccd570507d211d879ff29","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_linen/test_regression_issues_51_52.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_linen/test_regression_issues_51_52.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":13644,"artifact_hash":"sha256:671878843e509d50608c0bbbb654f594a082fe6d98fca6ec3972a34409fb2bb2","verifier":"a-root","branch":"codex/acceptance-2e5d","commit_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","tree_sha":"9f74c0aa84f8924a738ccd570507d211d879ff29","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","observed_at":"2026-08-30T11:26:11.123093Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
