---
id: f-nnx-grouped-yatconv-fails-for-every-spatial-dimension
kind: note
note_kind: finding
created: 2026-08-22T21:03:35Z
created_by: a-root
about: "[[t-01M0N7AQVHA9TCHD11Z9CDT9RH]]"
severity: major
---
# NNX grouped YatConv fails for every spatial dimension
Evidence: src/nmn/nnx/layers/conv/yat_conv.py constructs the patch-norm ones kernel with one output feature and passes feature_group_count to conv_general_dilated. JAX requires RHS output features divisible by group count. Reproduced on JAX 0.9.2 for 1D 2D and 3D with in_features=4 out_features=4 feature_group_count=2; every call raises ValueError that output feature dimension 1 is not a multiple of 2. No existing GitHub semantic duplicate was found. Expected: one patch-norm output per group, repeated across each group filters, with explicit reference and gradient tests.
