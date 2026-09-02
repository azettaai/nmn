---
id: f-nnx-yatconv-low-precision-distance-reconstruction-produces-invalid-scores
kind: note
note_kind: finding
created: 2026-08-22T21:08:27Z
created_by: a-root
about: "[[t-01M0N7AQVHA9TCHD11Z9CDT9RH]]"
severity: major
---
# NNX YatConv low-precision distance reconstruction produces invalid scores
Runtime-reproduced for 1D exact input/kernel: float16 seed 0 returned inf and bfloat16 seed 8 returned a negative score (-69). src/nmn/nnx/layers/conv/yat_conv.py reconstructs patch distance without maximum/clamp before division; likely affects all dimensional wrappers and requires testing transpose separately.
