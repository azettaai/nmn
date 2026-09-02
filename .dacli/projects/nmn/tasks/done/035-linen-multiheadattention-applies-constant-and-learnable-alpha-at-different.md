---
id: t-01M17A5NRQ0VEVMA1RQ229DHE5
kind: task
created: 2026-08-29T17:48:09Z
created_by: a-root
owner: a-root
github:
  issue: 52
  repo: azettaai/nmn
github_acceptance_import:
  issue: 52
  body_digest: sha256:e8a2ce62e632210410a722fe42f9432418a807c979556f10cf6ac8382aa96a7a
  actor: a-root
  imported_at: 2026-08-29T17:48:09Z
estimate: "{optimistic: 2, probable: 3, pessimistic: 5}"
---
# Linen MultiHeadAttention applies constant and learnable alpha at different stages
## Context
Adopted from GitHub issue #52.

## Summary

`nmn.linen.MultiHeadAttention` applies a constant alpha to the already-normalized attention output, while a learnable alpha is passed into the YAT score calculation before softmax/L1 normalization. Therefore a constant alpha and a learnable alpha set to the same value implement different functions.

## Affected code

- `src/nmn/linen/attention.py`
- `MultiHeadAttention.__call__`

## Reproduction

On commit `4f0298d`, use identical projection parameters and set both alpha variants to 2:

```python
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from nmn.linen.attention import MultiHeadAttention

x = jax.random.normal(jax.random.key(1), (1, 4, 8))
learnable = MultiHeadAttention(num_heads=2, use_alpha=True, constant_alpha=None)
constant = MultiHeadAttention(num_heads=2, use_alpha=True, constant_alpha=2.0)

variables = learnable.init(jax.random.key(0), x)
p = unfreeze(variables)
p["params"]["alpha"] = jnp.array([2.0])
learnable_variables = freeze(p)

p = unfreeze(learnable_variables)
del p["params"]["alpha"]
constant_variables = freeze(p)

y_learnable = learnable.apply(learnable_variables, x)
y_constant = constant.apply(constant_variables, x)
print(float(jnp.max(jnp.abs(y_learnable - y_constant))))
```

Observed maximum difference: `0.8976223`.

The learnable path calls `_nnx_yat_attention(..., alpha=alpha_val)`, which scales scores before normalization. The constant path calls it with `alpha=None` and then executes `x = x * _constant_alpha_value` after attention.

## Impact

Switching between fixed and trainable alpha changes attention semantics rather than only parameter trainability. Constant-alpha Linen attention also differs from the shared NNX score-scaling contract.

## Acceptance criteria

- [ ] Constant alpha is passed through the same score-scaling path as learnable alpha.
- [ ] Constant and learnable alpha produce matching outputs when their values and all projection parameters match.
- [ ] Tests cover softmax and L1 normalization, including alpha values other than 1.
- [ ] Documentation states the common alpha semantics.

## Acceptance
- [x] Constant alpha is passed through the same score-scaling path as learnable alpha.
- [x] Constant and learnable alpha produce matching outputs when their values and all projection parameters match.
- [x] Tests cover softmax and L1 normalization, including alpha values other than 1.
- [x] Documentation states the common alpha semantics.
## Log
- 2026-08-30T11:25:57Z accepted by a-root
- 2026-08-30T11:25:57Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_linen/test_regression_issues_51_52.py` (exit 0) in branch codex/acceptance-2e5d at 2e5d913 — proves that tree builds, not that the work is in trunk
- 2026-08-30T11:25:57Z deliverable: no dacli/035-linen-multiheadattention-applies-constant-and-learnable-alpha-at-different branch — nothing to check against master
- 2026-08-30T11:25:57Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_linen/test_regression_issues_51_52.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_linen/test_regression_issues_51_52.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":12985,"artifact_hash":"sha256:6c67f3f1db3ad4d1c7de871b3459c8769043628ad159f84a3835fb4861615f07","verifier":"a-root","branch":"codex/acceptance-2e5d","commit_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","tree_sha":"9f74c0aa84f8924a738ccd570507d211d879ff29","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_linen/test_regression_issues_51_52.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_linen/test_regression_issues_51_52.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":13470,"artifact_hash":"sha256:11a135ce9ef523c3665fd8794c181097e6d26566106239418003619433b67531","verifier":"a-root","branch":"codex/acceptance-2e5d","commit_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","tree_sha":"9f74c0aa84f8924a738ccd570507d211d879ff29","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"2e5d913f3c6f5becc3fe3a3f09942fc0cd3cc993","observed_at":"2026-08-30T11:25:57.17018Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
