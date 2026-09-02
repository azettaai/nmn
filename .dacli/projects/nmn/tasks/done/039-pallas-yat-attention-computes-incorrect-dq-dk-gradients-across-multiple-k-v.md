---
id: t-01M17A5PDS88H8WBKPAJB1G4FZ
kind: task
created: 2026-08-29T17:48:10Z
created_by: a-root
owner: a-root
github:
  issue: 48
  repo: azettaai/nmn
github_acceptance_import:
  issue: 48
  body_digest: sha256:4525af90493eadd44688d49181d8517e7135ba4f36d617062e11a20629b14e63
  actor: a-root
  imported_at: 2026-08-29T17:48:10Z
estimate: "{optimistic: 5, probable: 8, pessimistic: 13}"
---
# Pallas YAT attention computes incorrect dQ/dK gradients across multiple K/V tiles
## Context
Adopted from GitHub issue #48.

## Summary

The custom VJP in `pallas_yat_l1_attention` produces materially incorrect gradients for Q and K whenever attention spans more than one K/V tile. dV remains correct in the same reproduction.

## Affected code

- `src/nmn/nnx/layers/attention/pallas_yat_attention.py`
- `_yat_l1_bwd_dkv_kernel` and `_yat_l1_bwd_dq_kernel`

Each backward tile computes the normalization correction `sum(dW * W)` using only the current K/V tile. For L1-normalized attention, that correction must be accumulated across the full key sequence.

## Reproduction

On commit `4f0298d`, using JAX CPU Pallas interpret mode:

```python
import jax
import jax.numpy as jnp
from nmn.nnx.layers.attention.pallas_yat_attention import pallas_yat_l1_attention
from nmn.nnx.layers.attention.fused_yat_attention import fused_yat_l1_attention

q = jax.random.normal(jax.random.key(3), (1, 8, 1, 4))
k = jax.random.normal(jax.random.key(4), (1, 8, 1, 4))
v = jax.random.normal(jax.random.key(5), (1, 8, 1, 4))

def p_loss(q, k, v):
    return jnp.sum(pallas_yat_l1_attention(
        q, k, v, block_q=4, block_k=4, interpret=True
    ) ** 2)

def ref_loss(q, k, v):
    return jnp.sum(fused_yat_l1_attention(q, k, v) ** 2)

gp = jax.grad(p_loss, argnums=(0, 1, 2))(q, k, v)
gr = jax.grad(ref_loss, argnums=(0, 1, 2))(q, k, v)
print([float(jnp.max(jnp.abs(a - b))) for a, b in zip(gp, gr)])
```

Observed maximum errors:

```
dQ: 3.73876
dK: 3.93280
dV: 2.38e-7
```

Relative max errors are about 60% for dQ and 49% for dK.

The current gradient tests use a large tensor with a mean-reduced loss and an absolute tolerance of `5e-3`, which scales the wrong gradients enough for the test to pass.

## Impact

Training with the Pallas path can update Q/K projections using incorrect gradients on GPU/TPU even when forward outputs match the reference.

## Acceptance criteria

- [ ] dQ, dK, and dV match the fused reference for one and multiple K/V tiles.
- [ ] Tests include scale-independent relative/cosine checks or a sum-reduced loss that exposes this regression.
- [ ] Causal and non-causal backward parity are covered.
- [ ] The corrected kernel is validated in `interpret=True` and on a supported GPU or TPU backend.

## Acceptance
- [x] dQ, dK, and dV match the fused reference for one and multiple K/V tiles.
- [x] Tests include scale-independent relative/cosine checks or a sum-reduced loss that exposes this regression.
- [x] Causal and non-causal backward parity are covered.
- [x] The corrected kernel is validated in `interpret=True` and on a supported GPU or TPU backend.
## Log
- 2026-08-31T00:49:03Z accepted by a-root
- 2026-08-31T00:49:03Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest tests/test_nnx/test_pallas_yat_attention.py -q` (exit 0) in branch codex/acceptance-d6278 at f60a4d9 — proves that tree builds, not that the work is in trunk
- 2026-08-31T00:49:03Z deliverable: no dacli/039-pallas-yat-attention-computes-incorrect-dq-dk-gradients-across-multiple-k-v branch — nothing to check against master
- 2026-08-31T00:49:03Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=/private/tmp/nmn-pallas-tpu-fix/src /Users/tahabsn/.pixi/bin/python3 -m pytest /private/tmp/nmn-pallas-tpu-fix/tests/test_nnx/test_pallas_yat_attention.py -q","argv":["sh","-c","PYTHONPATH=/private/tmp/nmn-pallas-tpu-fix/src /Users/tahabsn/.pixi/bin/python3 -m pytest /private/tmp/nmn-pallas-tpu-fix/tests/test_nnx/test_pallas_yat_attention.py -q"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":19816,"artifact_hash":"sha256:4698aa1e3c8c50fc5136ba7e15032f03d59e7c0baa0452aaff96f10cc0d5d80c","verifier":"a-root","branch":"codex/acceptance-d6278","commit_sha":"d6278a2f8aa38a736e923f31b739ae6bd4b39a58","tree_sha":"96426f20f1935341a5929904239e705341dc96b0","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=/private/tmp/nmn-pallas-tpu-fix/src /Users/tahabsn/.pixi/bin/python3 -m pytest /private/tmp/nmn-pallas-tpu-fix/tests/test_nnx/test_pallas_yat_attention.py -q","argv":["sh","-c","PYTHONPATH=/private/tmp/nmn-pallas-tpu-fix/src /Users/tahabsn/.pixi/bin/python3 -m pytest /private/tmp/nmn-pallas-tpu-fix/tests/test_nnx/test_pallas_yat_attention.py -q"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":20017,"artifact_hash":"sha256:c8efc297a5c8bea9d99a82ea85135baaad3ad62a29adf2d7d72645c8a46443b3","verifier":"a-root","branch":"codex/acceptance-d6278","commit_sha":"d6278a2f8aa38a736e923f31b739ae6bd4b39a58","tree_sha":"96426f20f1935341a5929904239e705341dc96b0","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest tests/test_nnx/test_pallas_yat_attention.py -q","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest tests/test_nnx/test_pallas_yat_attention.py -q"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":20218,"artifact_hash":"sha256:5460157a62520c7767d643fd24254a02e3648249f573cc4513c37523d75a352c","verifier":"a-root","branch":"codex/acceptance-d6278","commit_sha":"f60a4d9f2469745f551970d19633cfe4a7a9e02f","tree_sha":"42aa5e1cff41ae9f69a69a42e39a7809a45cd4a5","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"f60a4d9f2469745f551970d19633cfe4a7a9e02f","observed_at":"2026-08-31T00:49:03.517669Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
