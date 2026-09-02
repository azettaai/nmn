---
id: d-use-cpu-jax-pallas-interpret-mode-for-reproducible-semantic-checks-and-reserve
kind: note
note_kind: decision
created: 2026-08-22T15:40:29Z
created_by: a-nmn-auditor-187qrm
about: "[[t-01M0N1M68973FWMQFG54QNSSCV]]"
---
# Use CPU JAX/Pallas interpret mode for reproducible semantic checks and reserve TPU lowering claims
## Chose
Use CPU JAX/Pallas interpret mode for reproducible semantic checks and reserve TPU lowering claims
## Rejected
Infer TPU correctness from source inspection alone
## Because
No TPU device is present; CPU JIT and Pallas interpret mode can confirm backend-independent math/gradient defects, while device-specific lowering and sharding require target-hardware evidence.
