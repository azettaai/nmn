---
id: f-keras-convolution-kernel-bank-auto-expansion-always-fails-on-fixed-shape
kind: note
note_kind: finding
created: 2026-08-22T21:20:06Z
created_by: a-root
about: "[[t-01M0N7AQVHA9TCHD11Z9CDT9RH]]"
severity: major
---
# Keras convolution kernel-bank auto-expansion always fails on fixed-shape variables
Runtime-reproduced on Keras JAX for YatConv1D and YatConvTranspose1D: create a tied bank of 3 filters, then request 5 filters with the same bank id; build raises ValueError because Variable.assign requires the original and replacement shapes to match. The same expansion code is duplicated across all 1D/2D/3D forward and transpose classes. The API/documentation advertises auto-expanding shared banks, but a Keras Variable cannot be resized by assign.
