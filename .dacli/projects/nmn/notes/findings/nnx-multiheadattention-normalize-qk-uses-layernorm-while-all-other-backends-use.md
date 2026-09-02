---
id: f-nnx-multiheadattention-normalize-qk-uses-layernorm-while-all-other-backends-use
kind: note
note_kind: finding
created: 2026-08-22T21:45:19Z
created_by: a-root
about: "[[t-01M0N7ATD4SKWBNH770NRGBY2Y]]"
severity: moderate
---
# NNX MultiHeadAttention normalize_qk uses LayerNorm while all other backends use L2 normalization
Source-verified parity defect: torch, Linen, Keras, TensorFlow, and MLX document and implement normalize_qk=True as per-head L2 normalization. NNX instead constructs learnable LayerNorm modules for Q and K, changing both values and parameter/state structure under the same public option. NNX's standalone normalize_qk function is L2, making its module internally inconsistent too.
