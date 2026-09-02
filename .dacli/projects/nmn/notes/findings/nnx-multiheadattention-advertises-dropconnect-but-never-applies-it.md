---
id: f-nnx-multiheadattention-advertises-dropconnect-but-never-applies-it
kind: note
note_kind: finding
created: 2026-08-22T21:49:34Z
created_by: a-root
about: "[[t-01M0N7ATD4SKWBNH770NRGBY2Y]]"
severity: major
---
# NNX MultiHeadAttention advertises DropConnect but never applies it
Runtime-reproduced: MultiHeadAttention(use_dropconnect=True,dropconnect_rate=0.5,dropout_rate=0) produces bit-identical training outputs with different dropout RNG streams. Source stores the flags and uses them only to require a deterministic decision; it never masks Q/K/V or output projection kernels. The option silently has no effect.
