---
id: d-use-runnable-torch-and-jax-family-dense-checks-with-source-plus-test-evidence
kind: note
note_kind: decision
created: 2026-08-22T21:11:26Z
created_by: a-nmn-auditor-qqf66n
about: "[[t-01M0N7AN7XAXWW4EN9R4DCFCRW]]"
---
# Use runnable Torch and JAX-family dense checks, with source-plus-test evidence for TensorFlow and MLX
## Chose
Use runnable Torch and JAX-family dense checks, with source-plus-test evidence for TensorFlow and MLX
## Rejected
Treat skipped or aborting backends as verified by import status
## Because
The mandated pixi environment runs Torch, NNX, Linen, and Keras/JAX reproductions; TensorFlow is absent and direct MLX pytest exits 134 headlessly, so implementation/test inspection is the strongest non-duplicative evidence available without target runtime support.
