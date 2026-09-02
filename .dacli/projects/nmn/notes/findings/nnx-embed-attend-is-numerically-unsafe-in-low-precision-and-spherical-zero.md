---
id: f-nnx-embed-attend-is-numerically-unsafe-in-low-precision-and-spherical-zero
kind: note
note_kind: finding
created: 2026-08-22T21:08:27Z
created_by: a-root
about: "[[t-01M0N7ATD4SKWBNH770NRGBY2Y]]"
severity: major
---
# NNX Embed attend is numerically unsafe in low precision and spherical zero vectors
Runtime-reproduced: exact-match float16 embedding/query returns inf at seed 0; bfloat16 returns a negative score (-1872) at seed 8 because reconstructed squared distance is not clamped. spherical=True with an all-zero query returns all NaNs because query and embedding norms are divided without an epsilon/clamp. Linen already guards normalization and distance, showing backend parity drift.
