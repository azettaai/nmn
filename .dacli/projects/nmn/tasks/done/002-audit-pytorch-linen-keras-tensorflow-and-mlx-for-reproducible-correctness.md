---
id: t-01M0N1M6AFM02BZN9S5FKKNQEN
kind: task
created: 2026-08-22T15:32:28Z
created_by: a-root
owner: a-root
priority: must
estimate: "{optimistic: 8, probable: 14, pessimistic: 21}"
---
# Audit PyTorch Linen Keras TensorFlow and MLX for reproducible correctness defects
## So that
All advertised NMN backends have evidence-backed correctness gaps tracked
## Acceptance
- [x] Each backend is reviewed for YatNMN, convolution, embedding, attention, dtype, lazy mode, and serialization behavior where implemented
- [x] Every confirmed defect includes file:line, minimal reproduction or decisive code path, impact, and proposed acceptance criteria
- [x] Candidates are checked against tests and existing GitHub issues before recommendation
## Log
- 2026-08-22T15:44:44Z claimed by a-nmn-auditor-55x57h
- 2026-08-22T15:55:04Z finding by a-nmn-auditor-55x57h: PyTorch mixed dtype attention fails before advertised compute promotion (event 01M0N2NKPGGQBAKV0CAPHDH1C4)
- 2026-08-22T15:55:04Z finding by a-nmn-auditor-55x57h: PyTorch low-precision conv and embedding can emit negative or infinite YAT scores (event 01M0N2NKQ5MC2JW2BSEHGRXJGZ)
- 2026-08-22T15:55:04Z finding by a-nmn-auditor-55x57h: Linen grouped YatConv1D 2D and 3D always fail XLA validation (event 01M0N2NKV00PKPKB768B4C975X)
- 2026-08-22T15:55:04Z finding by a-nmn-auditor-55x57h: Linen constant attention alpha scales values instead of attention scores (event 01M0N2NKVBBMF4PWSEMPETNM73)
- 2026-08-22T15:55:04Z finding by a-nmn-auditor-55x57h: Keras float16 and bfloat16 conv/embed emit infinite or negative YAT scores (event 01M0N2V2EP49MGCM4REWGVPEER)
- 2026-08-22T15:55:04Z finding by a-nmn-auditor-55x57h: Keras YatEmbed and MultiHeadYatAttention cannot deserialize without custom objects (event 01M0N2V2F671APBTYAD0JFMZPH)
- 2026-08-22T15:55:04Z finding by a-nmn-auditor-55x57h: TensorFlow grouped forward convolutions have incompatible patch-norm channel construction (event 01M0N2V2FPSYDA8HYDMCVQDX7A)
- 2026-08-22T15:55:04Z finding by a-nmn-auditor-55x57h: TensorFlow modules do not survive SavedModel as callable layers (event 01M0N2V2G3HTACJGT6REM5WM5K)
- 2026-08-22T15:55:04Z finding by a-nmn-auditor-55x57h: MLX fused YatNMN detaches learnable epsilon from autodiff (event 01M0N2V2GFFQ16EYDPZJSPFWFT)
- 2026-08-22T15:55:04Z finding by a-nmn-auditor-55x57h: nmn doctor aborts the process while probing MLX in headless environments (event 01M0N2V2GV2BJMH6JC5FXCH2PH)
- 2026-08-22T16:37:14Z completed by a-root
