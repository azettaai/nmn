---
id: f-rotaryyatattention-maclaurin-and-radial-performer-modes-silently-ignore-masks
kind: note
note_kind: finding
created: 2026-08-22T21:49:57Z
created_by: a-root
about: "[[t-01M0N7ATD4SKWBNH770NRGBY2Y]]"
severity: major
---
# RotaryYatAttention maclaurin and radial performer modes silently ignore masks
Runtime-reproduced for performer_kind='maclaurin' and 'radial': outputs with an all-True mask and an identity-only mask are bit-identical (max diff 0). These branches call the linear attention functions with only causal=self.causal and never pass or validate the public mask argument; SLAY and quadratic paths do pass masks. Padding and access-control masks are therefore silently bypassed.
