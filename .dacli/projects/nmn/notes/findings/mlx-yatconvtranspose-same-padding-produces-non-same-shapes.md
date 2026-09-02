---
id: f-mlx-yatconvtranspose-same-padding-produces-non-same-shapes
kind: note
note_kind: finding
created: 2026-08-22T21:19:21Z
created_by: a-root
about: "[[t-01M0N7AQVHA9TCHD11Z9CDT9RH]]"
severity: moderate
---
# MLX YatConvTranspose SAME padding produces non-SAME shapes
Source-decided because MLX aborts in this headless environment. _YatConvTransposeBase maps SAME to symmetric floor(((kernel-1)*dilation)/2) padding and leaves output_padding unchanged. By the transposed-convolution output formula, kernel=3/stride=2/default output_padding yields 2*input-1, and kernel=4/stride=1 yields input+1, not the SAME target. The source comment itself acknowledges the approximation is off for even kernels/stride, but the public padding='same' option gives no warning.
