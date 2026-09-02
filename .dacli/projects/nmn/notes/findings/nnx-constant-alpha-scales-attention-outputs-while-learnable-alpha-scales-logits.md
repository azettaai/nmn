---
id: f-nnx-constant-alpha-scales-attention-outputs-while-learnable-alpha-scales-logits
kind: note
note_kind: finding
created: 2026-08-22T21:44:31Z
created_by: a-root
about: "[[t-01M0N7ATD4SKWBNH770NRGBY2Y]]"
severity: major
---
# NNX constant alpha scales attention outputs while learnable alpha scales logits
Runtime-reproduced at alpha=sqrt(2): passing alpha to yat_attention (the learnable path) versus multiplying its output (the constant path used by MultiHeadAttention and RotaryYatAttention) differs by up to 1.38 on a tiny deterministic example. Pre-softmax scaling changes the distribution; post-attention multiplication only changes output magnitude. This is the same semantic class as open Linen issue #52 but affects distinct NNX modules and code paths.
