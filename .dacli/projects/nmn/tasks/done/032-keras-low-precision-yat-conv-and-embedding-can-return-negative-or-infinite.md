---
id: t-01M17A5NB28MX1Y4JQMG1XEJ4C
kind: task
created: 2026-08-29T17:48:09Z
created_by: a-root
owner: a-root
github:
  issue: 55
  repo: azettaai/nmn
github_acceptance_import:
  issue: 55
  body_digest: sha256:4e9f114857d3e4c05bb5302a78586bb2b96c4fad376bb2ec8af2700f8bc4f893
  actor: a-root
  imported_at: 2026-08-29T17:48:09Z
estimate: "{optimistic: 3, probable: 5, pessimistic: 8}"
depends_on: "[t-01M17A5N54KX8WK3NGECV5R935]"
---
# Keras low-precision YAT conv and embedding can return negative or infinite scores
## Context
Adopted from GitHub issue #55.

## Summary

Keras YAT convolution and embedding paths do not clamp cancellation-induced negative squared distances. Under the JAX backend, exact query/kernel matches in float16 and bfloat16 produce infinity or negative YAT scores.

## Affected code

- `src/nmn/keras/conv.py` (all forward and transpose convolution variants)
- `src/nmn/keras/embed.py`

## Reproduction

On commit `4f0298d`, Keras 3.14.1 with JAX 0.9.2:

```python
import jax
import jax.numpy as jnp
from nmn.keras import YatEmbed, YatConv1D

for dtype_name, dtype, seed in [
    ("float16", jnp.float16, 0),
    ("bfloat16", jnp.bfloat16, 8),
]:
    q = jax.random.normal(jax.random.key(seed), (1, 3), dtype=dtype)

    embed = YatEmbed(
        1, 3, use_alpha=False, epsilon=1e-5,
        dtype=dtype_name, embedding_initializer="zeros",
    )
    embed(jnp.array([0]))
    embed.embedding.assign(q)
    print(dtype_name, embed.attend(q))

    x = q.reshape(1, 3, 1)
    kernel = q.reshape(3, 1, 1)
    conv = YatConv1D(
        1, 3, use_bias=False, use_alpha=False,
        epsilon=1e-5, dtype=dtype_name,
        kernel_initializer="zeros",
    )
    conv(x)
    conv.kernel.assign(kernel)
    print(dtype_name, conv(x))
```

Observed for both embedding and convolution:

```
float16:  inf
bfloat16: -69
```

The bfloat16 computed distance is `-0.015625`.

## Impact

Supported Keras dtype policies can silently return invalid YAT values in convolution and tied-embedding output paths.

## Acceptance criteria

- [ ] Clamp distances to zero before epsilon for embedding and all six conv/transpose variants.
- [ ] JAX-backed float16/bfloat16 exact-match tests produce finite, non-negative outputs and gradients.
- [ ] Equivalent TensorFlow-backend tests cover the same dtype policies where supported.
- [ ] Low-precision outputs are compared with an fp32 reference on non-collision inputs.

## Acceptance
- [x] Clamp distances to zero before epsilon for embedding and all six conv/transpose variants.
- [x] JAX-backed float16/bfloat16 exact-match tests produce finite, non-negative outputs and gradients.
- [x] Equivalent TensorFlow-backend tests cover the same dtype policies where supported.
- [x] Low-precision outputs are compared with an fp32 reference on non-collision inputs.
## Log
- 2026-08-29T18:02:28Z dependency edit by a-root (event 01M17AZWGMSZYYKYVA3KGQN0M5)
- 2026-08-30T11:39:29Z accepted by a-root
- 2026-08-30T11:39:29Z verified by `KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py` (exit 0) in branch codex/acceptance-2a209 at 2a209b9 — proves that tree builds, not that the work is in trunk
- 2026-08-30T11:39:29Z deliverable: no dacli/032-keras-low-precision-yat-conv-and-embedding-can-return-negative-or-infinite branch — nothing to check against master
- 2026-08-30T11:39:29Z completed by a-root
## Verification Evidence
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":30088,"artifact_hash":"sha256:66a47fd7e23e694013b42cc0b9783f9b5b9c061d88bbe210a557217f4f0178c1","verifier":"a-root","branch":"codex/acceptance-2a209","commit_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","tree_sha":"eaff49d53d9a200fbf47f760a58c36d9a739e76a","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":27670,"artifact_hash":"sha256:eca108090eebb63bf0c606f54bec03af91aa25c78af3f8f3180bfc8c88c22e51","verifier":"a-root","branch":"codex/acceptance-2a209","commit_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","tree_sha":"eaff49d53d9a200fbf47f760a58c36d9a739e76a","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","observed_at":"2026-08-30T11:39:29.903719Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
