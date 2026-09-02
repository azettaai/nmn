---
id: f-nnx-yatconvtranspose-fails-whenever-transpose-kernel-is-enabled
kind: note
note_kind: finding
created: 2026-08-22T21:17:51Z
created_by: a-root
about: "[[t-01M0N7AQVHA9TCHD11Z9CDT9RH]]"
severity: major
---
# NNX YatConvTranspose fails whenever transpose_kernel is enabled
Runtime-reproduced for 1D/2D/3D YatConvTranspose with in_features=2, out_features=3, transpose_kernel=True: every call raises a conv_general_dilated feature mismatch (2 != 1). The primary transposed convolution accepts the transposed kernel layout, but the patch-squared-sum ones kernel is shaped with a singleton output axis that becomes the rhs input-feature axis under transpose_kernel=True.
