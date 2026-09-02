---
id: f-nnx-learnable-epsilon-collapses-to-zero-in-low-precision-parameter-dtypes
kind: note
note_kind: finding
created: 2026-08-22T21:08:27Z
created_by: a-root
about: "[[t-01M0N7AN7XAXWW4EN9R4DCFCRW]]"
severity: major
---
# NNX learnable epsilon collapses to zero in low-precision parameter dtypes
Runtime-reproduced with /Users/tahabsn/.pixi/bin/python3 and PYTHONPATH=src: param_dtype=float16 or bfloat16 with epsilon=1e-5 initializes raw epsilon to -inf because exp(epsilon) rounds to 1 before subtracting 1; softplus(float32(-inf)) is 0. Affected source patterns exist in dense YatNMN, Embed, YatConv, YatConvTranspose, MultiHeadAttention, and RotaryYatAttention. float32 initializes to -11.511568 and remains finite.
