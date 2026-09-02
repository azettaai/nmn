---
id: f-rotaryyatattention-incremental-decode-uses-rope-position-zero-for-every-query
kind: note
note_kind: finding
created: 2026-08-22T21:46:29Z
created_by: a-root
about: "[[t-01M0N7ATD4SKWBNH770NRGBY2Y]]"
severity: major
---
# RotaryYatAttention incremental decode uses RoPE position zero for every query token
Runtime-reproduced after locally initializing caches with nnx.data to bypass the separate init_cache blocker: a length-4 full causal forward and token-by-token decode with identical parameters agree at token 0, then differ by 0.99, 0.78, and 1.51 max per step. Decode sets position_offset=0 before calling rotary attention over a one-token query and full key cache, so every new query is rotated as position 0 instead of cur_index. Keys need positions 0..cache length while the query needs its current index.
