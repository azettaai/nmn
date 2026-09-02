---
id: t-01M0N7AN7XAXWW4EN9R4DCFCRW
kind: task
created: 2026-08-22T17:12:07Z
created_by: a-root
owner: a-root
estimate: "{optimistic: 8, probable: 15, pessimistic: 21}"
---
# Deep-audit dense YatNMN parity gradients dtypes and lazy semantics
## Acceptance
- [x] Every backend dense implementation is compared to an explicit reference across bias alpha epsilon spherical normalized and lazy modes
- [x] Forward and gradient parity cover ranks edge shapes collision inputs and supported low precision dtypes
- [x] Findings include reproduction impact test gap and semantic GitHub deduplication
## Log
- 2026-08-22T17:12:59Z claimed by a-nmn-auditor-rys7wb
- 2026-08-22T20:59:47Z finding by a-nmn-auditor-rys7wb: PyTorch float16 dense YatNMN overflows at exact collisions despite distance clamping (event 01M0N7GYCVDMRW313J181D7CHP)
- 2026-08-22T21:13:27Z finding by a-nmn-auditor-qqf66n: Spherical dense shortcut miscomputes zero-vector distance and JAX gradients become NaN (event 01M0NMT9ARQGBD39EQG2703VFT)
- 2026-08-22T21:13:27Z finding by a-nmn-auditor-qqf66n: NNX freeze_kernel false silently disables lazy true unlike every other backend (event 01M0NMT9B7TT5CQ5GJP1MG6K1E)
- 2026-08-22T21:13:27Z finding by a-nmn-auditor-qqf66n: NNX fused dense custom VJP catastrophically disagrees near collisions (event 01M0NMWQGWKDFKVFD57SHCZF93)
- 2026-08-22T21:13:27Z finding by a-nmn-auditor-qqf66n: PyTorch lazy tied kernel globally freezes eager peer layers (event 01M0NMXQZQCSXGH5Y8VGHVXE5T)
- 2026-08-22T21:13:27Z finding by a-nmn-auditor-qqf66n: PyTorch tied-bank construction reinitializes existing layers (event 01M0NMYFKV1ZVNMQ5W7QRDAWGX)
- 2026-08-22T21:13:40Z accepted by a-root
- 2026-08-22T21:13:40Z closed WITHOUT verification — no --verify command was given
- 2026-08-22T21:13:40Z deliverable: no dacli/003-deep-audit-dense-yatnmn-parity-gradients-dtypes-and-lazy-semantics branch — nothing to check against master
- 2026-08-22T21:13:40Z completed by a-root
