---
id: t-01M0N1M68973FWMQFG54QNSSCV
kind: task
created: 2026-08-22T15:32:28Z
created_by: a-root
owner: a-root
priority: must
estimate: "{optimistic: 5, probable: 10, pessimistic: 18}"
---
# Audit TPU and JAX execution paths for reproducible correctness defects
## So that
TPU users do not encounter silent numerical, lowering, sharding, or gradient failures
## Acceptance
- [x] Every confirmed defect includes backend, file:line, triggering configuration, expected behavior, and observed behavior
- [x] Pallas, fused YatNMN, attention, BF16/mixed precision, sharding, and TPU examples are each reviewed
- [x] Candidates are checked against tests and existing GitHub issues before recommendation
## Log
- 2026-08-22T15:35:57Z claimed by a-nmn-auditor-5r1sha
- 2026-08-22T15:44:04Z finding by a-nmn-auditor-187qrm: GitHub issue deduplication unavailable in audit sandbox (event 01M0N22W335AFM7QKEG6YZ8ANX)
- 2026-08-22T16:37:08Z completed by a-root
