---
id: d-route-jax-dense-collision-gradient-evidence-into-the-existing-low-precision
kind: note
note_kind: decision
created: 2026-08-22T21:12:51Z
created_by: a-nmn-auditor-qqf66n
about: "[[t-01M0N7AN7XAXWW4EN9R4DCFCRW]]"
---
# Route JAX dense collision gradient evidence into the existing low-precision finding
## Chose
Route JAX dense collision gradient evidence into the existing low-precision finding
## Rejected
File a new backend-specific collision-gradient defect
## Because
Pixi/JAX exact-collision repro with w=[0.25,-0.5,0.75], bias and alpha disabled: float16 Linen/NNX/Keras all return inf and non-finite input gradients; bfloat16 Linen and Keras return finite 76288 but an all-zero input gradient, while NNX float32-score computation returns finite 76800 and nonzero [43776,-87552,131072]. The existing PyTorch dense low-precision finding already requires cross-backend float16/bfloat16 collision forward and gradient reference tests, so this evidence expands its impact without creating a semantic duplicate.
