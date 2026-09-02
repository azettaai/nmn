---
id: f-softmax-yat-attention-leaks-or-returns-nans-for-fully-masked-query-rows
kind: note
note_kind: finding
created: 2026-08-22T21:43:30Z
created_by: a-root
about: "[[t-01M0N7ATD4SKWBNH770NRGBY2Y]]"
severity: major
---
# Softmax YAT attention leaks or returns NaNs for fully masked query rows
Runtime-reproduced with q_len=1, kv_len=2, mask all False. PyTorch yat_attention_weights returns [NaN, NaN] because it softmaxes all -inf. Keras JAX and NNX softmax return [0.5,0.5] because finite large-negative sentinels become a uniform distribution, so the supposedly masked query attends to values. Linen delegates to NNX; TensorFlow/MLX source use the same finite-sentinel pattern. NNX l1/softermax happen to return zeros. A post-normalization mask/renormalization or explicit all-masked-row handling is required.
