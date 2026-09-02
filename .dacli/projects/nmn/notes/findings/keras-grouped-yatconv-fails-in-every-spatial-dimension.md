---
id: f-keras-grouped-yatconv-fails-in-every-spatial-dimension
kind: note
note_kind: finding
created: 2026-08-22T21:15:59Z
created_by: a-root
about: "[[t-01M0N7AQVHA9TCHD11Z9CDT9RH]]"
severity: major
---
# Keras grouped YatConv fails in every spatial dimension
Runtime-reproduced with Keras JAX backend for YatConv1D/2D/3D, input channels=4, filters=4, groups=2: every call raises conv_general_dilated rhs output feature dimension size must be a multiple of feature_group_count, but 1 is not a multiple of 2. The patch-norm helper kernel has one output channel while ops.conv infers grouped convolution; existing Keras tests do not cover groups>1.
