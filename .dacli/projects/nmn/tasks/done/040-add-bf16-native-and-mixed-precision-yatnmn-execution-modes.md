---
id: t-01M17A5PJABDE71M0YXR39TSXA
kind: task
created: 2026-08-29T17:48:10Z
created_by: a-root
owner: a-root
github:
  issue: 47
  repo: azettaai/nmn
github_acceptance_import:
  issue: 47
  body_digest: sha256:f170f9c52334036d36ffd432a2ce16f12ba7558a8e10198c4d96343777cbecf2
  actor: a-root
  imported_at: 2026-08-29T17:48:10Z
estimate: "{optimistic: 5, probable: 8, pessimistic: 13}"
---
# Add BF16-native and mixed-precision YatNMN execution modes
## Context
Adopted from GitHub issue #47.

## Motivation

`YatNMN` currently promotes the complete input and kernel operands to FP32 in both the standard and fused Flax NNX paths:

```python
inputs_f32 = inputs.astype(jnp.float32)
kernel_f32 = kernel.astype(jnp.float32)
```

This is a safe reference implementation, but it prevents an actually BF16-native Yat FFN and may add conversion traffic or FP32 operand materialization on TPU. This becomes important when `YatNMN` replaces the FFN in a transformer, where the operands are much larger than in the existing small examples.

The accompanying comment says that squaring the score causes BF16 overflow. BF16 has the same exponent width and essentially the same dynamic range as FP32, so overflow is not normally the distinguishing failure mode. The important BF16 risks here are instead:

1. the seven-bit mantissa;
2. cancellation in
   `||x||² + ||w||² - 2 xᵀw` when `x` and `w` are close;
3. error amplification by the reciprocal denominator; and
4. the `D^{-2}` dependence appearing in derivatives of the rational score.

Full FP32 evaluation addresses all four, but it is more conservative than necessary for every workload.

## Proposed design

Preserve the current FP32 implementation as the default/reference path, and add explicit opt-in numerical modes, for example:

```python
YatNMN(
    ...,
    compute_mode="fp32",       # "fp32" | "mixed" | "bf16"
    distance_floor=0.0,
)
```

The exact API is open for discussion; the important point is that `dtype=jnp.bfloat16` currently does not imply BF16 Yat-score evaluation because the layer overrides it internally.

### 1. `fp32`: reference behavior

Keep the existing implementation and cross-framework FP32 parity unchanged.

### 2. `mixed`: recommended TPU path

Keep parameters and activations as BF16 and pass BF16 operands directly to `dot_general`, while requesting FP32 accumulation/output with `preferred_element_type=jnp.float32` when supported. Norm reductions can similarly use an FP32 accumulator without first materializing complete FP32 copies of the input and kernel:

```python
dot = lax.dot_general(
    x_bf16,
    w_bf16,
    dimension_numbers,
    precision=precision,
    preferred_element_type=jnp.float32,
)
x_sq = jnp.sum(x_bf16 * x_bf16, axis=-1, keepdims=True, dtype=jnp.float32)
w_sq = jnp.sum(w_bf16 * w_bf16, axis=0, keepdims=True, dtype=jnp.float32)
```

This should retain the numerically sensitive accumulations while avoiding the unconditional full-operand casts. Whether XLA already fuses the current conversions should be verified from lowered HLO and TPU profiles rather than assumed.

### 3. `bf16`: experimental strict-BF16 path

Use a dimension-scaled form so intermediates stay near unit scale. For input width `d`, define

```text
m_j  = mean_i(x_i w_ij)
r_x  = mean_i(x_i²)
r_wj = mean_i(w_ij²)
r_j  = max(r_x + r_wj - 2 m_j, δ_j)
```

Then compute

```text
y_j = d (m_j + b_j/d)² / (r_j + ε/d)
```

followed by the existing alpha scaling. Before the distance floor and finite-precision rounding, this is algebraically identical to

```text
(xᵀw_j + b_j)² / (||x - w_j||² + ε).
```

`δ_j` could be an absolute floor or a relative floor such as

```text
max(abs_floor, rel_floor * (r_x + r_wj)).
```

The spherical path needs the analogous protection around `2 - 2 xᵀw`. The fused/custom-VJP implementation should differentiate the clamped forward expression consistently rather than silently using an unrelated gradient rule.

Input RMS normalization, controlled kernel norms, and global gradient clipping remain useful training-level safeguards, but they should not be required to make the layer numerically defined.

## Validation plan

Please benchmark all three modes rather than replacing the reference implementation immediately:

- input widths from 128 to 4096;
- random inputs plus adversarial near-collision cases `x ≈ w`;
- several input/kernel scales and epsilon values from `1e-5` to `1`;
- forward absolute/relative error against FP32;
- input/kernel gradient error and gradient cosine similarity;
- NaN/Inf incidence and maximum activation/gradient magnitude;
- standard versus fused path parity;
- lowered HLO, peak memory, compile time, and steady-state step time on TPU;
- a 500–1000-step transformer FFN training smoke test.

Suggested acceptance criteria:

1. `compute_mode="fp32"` remains numerically equivalent to the current release.
2. `mixed` avoids explicit full-array FP32 operand casts and stays close to the FP32 forward and backward reference.
3. `bf16` remains finite across the declared scale/epsilon test domain, with its error envelope documented.
4. Precision behavior is documented for both standard and fused NNX paths.

This would let users choose between strict reference accuracy, a likely best TPU performance/accuracy compromise, and an experimental all-BF16 implementation without changing the current default silently.

## Acceptance
- [x] `compute_mode="fp32"` remains numerically equivalent to the prior reference path.
- [x] `mixed` avoids explicit full-array FP32 operand casts and has forward and backward parity with the FP32 reference.
- [x] `bf16` remains finite across the declared scale and epsilon test domain, with its error envelope covered by tests.
- [x] Precision behavior is documented for both standard and fused NNX paths.
## Log
- 2026-08-29T18:11:49Z a-root: PR opened: https://github.com/azettaai/nmn/pull/80 (event 01M17AY7B0NZDGY6BP48APXB7M)
- 2026-08-29T18:13:25Z completion requested by a-root; PR landing state unlanded on master
- 2026-08-29T18:23:03Z accepted by a-root
- 2026-08-29T18:23:03Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_yatnmn_precision.py` (exit 0) in commit 9894971 — proves that tree builds, not that the work is in trunk
- 2026-08-29T18:23:03Z deliverable: dacli/040-add-bf16-native-and-mixed-precision-yatnmn-execution-modes exists but is NOT in master — closed anyway
- 2026-08-29T18:23:03Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_yatnmn_precision.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_yatnmn_precision.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":16472,"artifact_hash":"sha256:f21eaace4811b707980a91be4ac9f4555776b59b7afaf120f4d0b76d1cb5bc1f","verifier":"a-root","branch":"dacli/040-add-bf16-native-and-mixed-precision-yatnmn-execution-modes","commit_sha":"4635a545a7096db0f04e81daa8f91db46c883a07","tree_sha":"05776a08a1f9a0373b63b9bd6378c03ed7fa03c1","runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_yatnmn_precision.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_yatnmn_precision.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn/.dacli/worktrees/accept-47","exit_code":0,"duration_ms":27481,"artifact_hash":"sha256:e1c02a4cc431e3c41692d5467e22a9ace036c9a1ea3d0afab001f4a3fcb213c6","verifier":"a-root","branch":"","commit_sha":"98949714a1852ad90416f16f5083a45c68f6c7ce","tree_sha":"05776a08a1f9a0373b63b9bd6378c03ed7fa03c1","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"98949714a1852ad90416f16f5083a45c68f6c7ce","observed_at":"2026-08-29T18:23:03.857884Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
