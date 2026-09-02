---
id: t-01M17A5P8V2NMFAAKA4CAKS4A5
kind: task
created: 2026-08-29T17:48:10Z
created_by: a-root
owner: a-root
github:
  issue: 49
  repo: azettaai/nmn
github_acceptance_import:
  issue: 49
  body_digest: sha256:5c10e579effdda543e7d3344f00371ae5b81e1dfb8bf4a1fb782951a9833ab37
  actor: a-root
  imported_at: 2026-08-29T17:48:10Z
estimate: "{optimistic: 5, probable: 8, pessimistic: 13}"
depends_on: "[t-01M17A5PDS88H8WBKPAJB1G4FZ]"
---
# Pallas YAT attention does not mask ragged K/V tiles or bound causal cross-attention
## Context
Adopted from GitHub issue #49.

## Summary

`pallas_yat_l1_attention` silently produces incorrect forward outputs when `kv_len` is not divisible by `block_k`. Causal cross-attention also reads beyond the available K/V blocks when `q_len > kv_len`.

The public docstring says the tile sizes must divide the sequence lengths, but the implementation uses `pl.cdiv` and does not validate that precondition. Out-of-bounds/padded tile elements are not masked from scores and normalization. The causal loop bound is also not capped by the available number of K/V tiles.

## Affected code

- `src/nmn/nnx/layers/attention/pallas_yat_attention.py`
- Forward and backward tiled loops

## Reproduction

On commit `4f0298d`, CPU Pallas interpret mode:

```python
import jax
import jax.numpy as jnp
from nmn.nnx.layers.attention.pallas_yat_attention import pallas_yat_l1_attention
from nmn.nnx.layers.attention.fused_yat_attention import fused_yat_l1_attention

def compare(q_len, kv_len, causal=False):
    q = jax.random.normal(jax.random.key(0), (1, q_len, 2, 4))
    k = jax.random.normal(jax.random.key(1), (1, kv_len, 2, 4))
    v = jax.random.normal(jax.random.key(2), (1, kv_len, 2, 4))
    mask = None
    if causal:
        mask = jnp.arange(q_len)[:, None] >= jnp.arange(kv_len)[None, :]
    out = pallas_yat_l1_attention(
        q, k, v, causal=causal, block_q=4, block_k=4, interpret=True
    )
    ref = fused_yat_l1_attention(q, k, v, mask=mask)
    return float(jnp.max(jnp.abs(out - ref)))

print(compare(8, 10, causal=False))
print(compare(12, 8, causal=True))
```

Observed:

```
ragged kv_len: 0.365541 max error
causal q_len > kv_len: 0.267788 max error
```

Aligned self-attention in the same setup matches within approximately `2.4e-7`.

## Impact

Variable-length batches and causal cross-attention can silently return wrong values on the Pallas GPU/TPU path.

## Acceptance criteria

- [ ] Partial K/V and Q tiles are masked in forward and backward, or unsupported shapes are rejected before kernel launch.
- [ ] Causal loop bounds never exceed the available K/V blocks.
- [ ] Forward and gradient parity tests cover ragged Q, ragged K/V, `q_len < kv_len`, and `q_len > kv_len`.
- [ ] Tests run in `interpret=True` and on a supported GPU or TPU backend.

## Acceptance
- [x] Partial K/V and Q tiles are masked in forward and backward, or unsupported shapes are rejected before kernel launch.
- [x] Causal loop bounds never exceed the available K/V blocks.
- [x] Forward and gradient parity tests cover ragged Q, ragged K/V, `q_len < kv_len`, and `q_len > kv_len`.
- [x] Tests run in `interpret=True` and on a supported GPU or TPU backend.
## Log
- 2026-08-29T18:02:31Z dependency edit by a-root (event 01M17AZZCEDT5KE77R533C1S3R)
- 2026-08-31T00:48:36Z accepted by a-root
- 2026-08-31T00:48:36Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest tests/test_nnx/test_pallas_yat_attention.py -q` (exit 0) in branch codex/acceptance-d6278 at f60a4d9 — proves that tree builds, not that the work is in trunk
- 2026-08-31T00:48:36Z deliverable: no dacli/038-pallas-yat-attention-does-not-mask-ragged-k-v-tiles-or-bound-causal-cross branch — nothing to check against master
- 2026-08-31T00:48:36Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=/private/tmp/nmn-pallas-tpu-fix/src /Users/tahabsn/.pixi/bin/python3 -m pytest /private/tmp/nmn-pallas-tpu-fix/tests/test_nnx/test_pallas_yat_attention.py -q","argv":["sh","-c","PYTHONPATH=/private/tmp/nmn-pallas-tpu-fix/src /Users/tahabsn/.pixi/bin/python3 -m pytest /private/tmp/nmn-pallas-tpu-fix/tests/test_nnx/test_pallas_yat_attention.py -q"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":19492,"artifact_hash":"sha256:5c0774b3796ee5d9610982c0b83a6e789c584283a9f87cb985219f675b5e1794","verifier":"a-root","branch":"codex/acceptance-d6278","commit_sha":"d6278a2f8aa38a736e923f31b739ae6bd4b39a58","tree_sha":"96426f20f1935341a5929904239e705341dc96b0","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest tests/test_nnx/test_pallas_yat_attention.py -q","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest tests/test_nnx/test_pallas_yat_attention.py -q"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":19307,"artifact_hash":"sha256:2e584be11668855c61a0523265ab65ecf80535e3a9ab9b70d8276762f7762664","verifier":"a-root","branch":"codex/acceptance-d6278","commit_sha":"f60a4d9f2469745f551970d19633cfe4a7a9e02f","tree_sha":"42aa5e1cff41ae9f69a69a42e39a7809a45cd4a5","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"f60a4d9f2469745f551970d19633cfe4a7a9e02f","observed_at":"2026-08-31T00:48:36.51857Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
