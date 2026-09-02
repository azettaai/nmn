---
id: d-extend-the-existing-dense-low-precision-finding-instead-of-filing-backend
kind: note
note_kind: decision
created: 2026-08-22T21:11:26Z
created_by: a-nmn-auditor-qqf66n
about: "[[t-01M0N7AN7XAXWW4EN9R4DCFCRW]]"
---
# Extend the existing dense low-precision finding instead of filing backend duplicates
## Chose
Extend the existing dense low-precision finding instead of filing backend duplicates
## Rejected
Create separate float16 collision findings for Linen, NNX, Keras, TF, and MLX
## Because
The sibling PyTorch dense collision finding already identifies the shared dense output-range contract and explicitly asks for cross-backend collision tests; separate notes would be semantic duplicates. This audit focuses new findings on distinct spherical, fused-gradient, alias, and kernel-bank causes.
