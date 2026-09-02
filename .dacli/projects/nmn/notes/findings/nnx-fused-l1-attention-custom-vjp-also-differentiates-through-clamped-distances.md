---
id: f-nnx-fused-l1-attention-custom-vjp-also-differentiates-through-clamped-distances
kind: note
note_kind: finding
created: 2026-08-22T21:48:52Z
created_by: a-root
about: "[[t-01M0N7AX4ADW74CAW034F73WJ0]]"
severity: major
---
# NNX fused L1 attention custom VJP also differentiates through clamped distances
Runtime-reproduced against an unfused autodiff reference with q and two nearly-colliding keys (width 64, scale 10): forwards are equal but the fused gradient has up to 6.8% relative error across a 200-seed search. _fused_yat_l1_attn_bwd forms g_dist for every entry and never multiplies by the derivative of maximum(raw_dist,0), the same root cause as the dense optimized VJP finding. One GitHub issue can cover both custom VJPs if acceptance names both paths.
