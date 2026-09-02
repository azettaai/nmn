---
id: 01M0NMT9ARQGBD39EQG2703VFT
kind: event
schema_version: 1
event_kind: finding
created: 2026-08-22T21:07:51Z
created_by: a-nmn-auditor-qqf66n
about: "[[t-01M0N7AN7XAXWW4EN9R4DCFCRW]]"
origin: agent
applied: true
checksum: sha256:6a8c5a0b535bab7a6b5bc623c17dcf16738cb682e2b73d392e925c1ab4d2d30e
---
Spherical dense shortcut miscomputes zero-vector distance and JAX gradients become NaN

Reproduction with /Users/tahabsn/.pixi/bin/python3 on torch 2.11 and JAX-backed Linen/NNX/Keras: spherical YatNMN with x=zeros((1,3)), unit kernel [1,0,0], bias=1, alpha disabled returns 0.4999975, while the explicit reference that first applies the implementation normalization and then computes sum((x_hat-w_hat)^2) returns 0.9999900. Torch reproduces at ranks 1, 2, and 3; Linen, NNX, and Keras input gradients at x=0 are NaN because jnp.linalg.norm is differentiated at zero. Root cause: src/nmn/torch/nmn/yat_nmn.py:343-366, src/nmn/nnx/layers/nmn.py HEAD lines 372-408, src/nmn/linen/nmn.py:182-211, src/nmn/keras/nmn.py:186-204, src/nmn/tf/nmn.py:210-232, and src/nmn/mlx/nmn.py:250-271 normalize with norm+1e-8 (so zero stays zero) but then assume both operands have norm exactly one and substitute distance=2-2*dot. The same invalid unit-norm assumption applies to zero kernels in weight_normalized mode. Impact: biased spherical dense outputs are wrong by nearly 2x at zero/near-zero inputs and JAX-family training can emit NaN gradients on padded or dead features. Existing dense formula tests use random nonzero inputs and their spherical references repeat the same shortcut (for example tests/test_mlx/test_yat_math_validation.py:36-52); no zero-vector direct-distance oracle exists. Acceptance: compare every backend to sum((x_hat-w_hat)^2) on zero and near-zero inputs/kernels across ranks, make forward and input/kernel gradients finite and reference-equivalent, including jit/low precision. Open local task listing has no semantic duplicate; GitHub issue inspection failed because api.github.com is unreachable, so remote dedup remains unverified.
