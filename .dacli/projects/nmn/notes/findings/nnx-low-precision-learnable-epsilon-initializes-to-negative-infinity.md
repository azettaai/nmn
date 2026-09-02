---
id: f-nnx-low-precision-learnable-epsilon-initializes-to-negative-infinity
kind: note
note_kind: finding
created: 2026-08-22T21:00:52Z
created_by: a-root
about: "[[t-01M0N7AN7XAXWW4EN9R4DCFCRW]]"
severity: major
---
# NNX low-precision learnable epsilon initializes to negative infinity
Evidence: src/nmn/nnx/layers/nmn.py lines 326-329 evaluates exp(epsilon)-1 directly in param_dtype. With the default epsilon 1e-5 and param_dtype float16 or bfloat16, exp rounds to 1, raw epsilon becomes -inf, and softplus is exactly zero. Exact input/kernel collisions then return inf; float32 initializes to -11.511568 and remains finite. Reproduced on JAX 0.9.2 for both float16 and bfloat16. Expected: inverse-softplus initialization computed stably in sufficient precision and cast afterward. GitHub search found no semantic duplicate.
