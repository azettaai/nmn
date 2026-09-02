---
id: f-nnx-optimized-fused-fp32-vjp-ignores-the-distance-clamp-derivative
kind: note
note_kind: finding
created: 2026-08-22T21:11:41Z
created_by: a-root
about: "[[t-01M0N7AN7XAXWW4EN9R4DCFCRW]]"
severity: major
---
# NNX optimized fused fp32 VJP ignores the distance clamp derivative
Runtime-reproduced directly against _yat_value with width 64 float32, random seed 2, scale 10, exact x==kernel: reconstructed raw distance is -0.0009765625 and both forwards use max(raw,0)+epsilon, but standard input-gradient norm is 1.12576421888e11 while _fused_yat_fp32 gradient norm is exactly 0 (relative error 1). The analytical VJP always differentiates the raw distance and omits the max clamp mask; standard autodiff correctly zeros the distance derivative when raw<0. The optimized path is selected for default fp32 with distance_floor=0.
