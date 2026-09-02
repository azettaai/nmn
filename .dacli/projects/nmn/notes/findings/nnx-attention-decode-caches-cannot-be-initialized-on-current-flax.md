---
id: f-nnx-attention-decode-caches-cannot-be-initialized-on-current-flax
kind: note
note_kind: finding
created: 2026-08-22T21:46:09Z
created_by: a-root
about: "[[t-01M0N7ATD4SKWBNH770NRGBY2Y]]"
severity: major
---
# NNX attention decode caches cannot be initialized on current Flax
Runtime-reproduced with Flax 0.12/JAX 0.9.2: MultiHeadAttention.init_cache((1,4,8)) and RotaryYatAttention.init_cache(1,4) both raise ValueError because __init__ made cached_key/cached_value/cache_index static None attributes and init_cache later assigns nnx.Cache data variables. Flax refuses changing attribute status without nnx.data. No incremental decoding can start. After fixing initialization, rotary decode parity should also test that the single-token query uses cur_index as its RoPE position rather than the current position_offset=0.
