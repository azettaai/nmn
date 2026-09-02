---
id: 01M0N2NKVBBMF4PWSEMPETNM73
kind: event
schema_version: 1
event_kind: finding
created: 2026-08-22T15:50:43Z
created_by: a-nmn-auditor-55x57h
about: "[[t-01M0N1M6AFM02BZN9S5FKKNQEN]]"
origin: agent
applied: true
checksum: sha256:14afabbd1eb353ffe23c5f65c5cf2dfc5ab109c3b174f5639fd1dcec8a363a4d
---
Linen constant attention alpha scales values instead of attention scores

Evidence: src/nmn/linen/attention.py:260-281 passes learnable alpha into _nnx_yat_attention where it scales pre-softmax scores, but lines 283-285 multiply the post-attention value output for constant_alpha. CPU repro with identical bias-free projection params, num_heads=2 and alpha=2: learnable-alpha and constant-alpha outputs differ by max abs 0.9158078. This contradicts the attention API definition in src/nmn/nnx/layers/attention/yat_attention.py:74-76 and PyTorch/Keras behavior. Impact: swapping learnable alpha for an equivalent fixed value changes the attention distribution and cross-backend results. Existing tests/test_linen/test_attention.py:72-77 checks only finiteness. Dedup: full GitHub list unavailable and targeted public search found no match. Acceptance: pass constant alpha into score normalization at the same location as learnable alpha; with shared params constant_alpha=a must equal learnable alpha fixed to a, including jit and serialization tree round-trips.
