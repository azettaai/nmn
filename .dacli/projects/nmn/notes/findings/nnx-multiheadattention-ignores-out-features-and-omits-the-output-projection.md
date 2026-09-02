---
id: f-nnx-multiheadattention-ignores-out-features-and-omits-the-output-projection
kind: note
note_kind: finding
created: 2026-08-22T21:45:19Z
created_by: a-root
about: "[[t-01M0N7ATD4SKWBNH770NRGBY2Y]]"
severity: major
---
# NNX MultiHeadAttention ignores out_features and omits the output projection
Runtime-reproduced: MultiHeadAttention(num_heads=2,in_features=8,qkv_features=8,out_features=3,decode=False) returns shape (1,4,8), not (1,4,3). Source stores out_features/out_kernel_init/out_bias_init but creates only Q/K/V projections and returns the reshaped attention result without an output projection. Existing tests set out_features equal to qkv_features, hiding the defect.
