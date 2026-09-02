---
id: f-keras-convolution-shape-inference-ignores-dilation-in-all-forward-and-transpose
kind: note
note_kind: finding
created: 2026-08-22T21:17:12Z
created_by: a-root
about: "[[t-01M0N7AQVHA9TCHD11Z9CDT9RH]]"
severity: moderate
---
# Keras convolution shape inference ignores dilation in all forward and transpose variants
Runtime-reproduced on Keras JAX for 1D/2D/3D forward and transpose layers with kernel_size=3, dilation_rate=2, padding=valid. Forward YatConv1D declares length 7 for input 9 but returns 5; transpose declares length 7 for input 5 but returns 9. Equivalent mismatches occur on every spatial axis in 2D/3D. All six custom compute_output_shape methods use raw kernel_size and omit the effective dilated kernel size.
