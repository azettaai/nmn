---
id: d-use-runnable-backend-checks-where-imports-succeed-and-source-plus-test-evidence
kind: note
note_kind: decision
created: 2026-08-22T15:46:00Z
created_by: a-nmn-auditor-55x57h
about: "[[t-01M0N1M6AFM02BZN9S5FKKNQEN]]"
---
# Use runnable backend checks where imports succeed and source-plus-test evidence where dependencies are unavailable
## Chose
Use runnable backend checks where imports succeed and source-plus-test evidence where dependencies are unavailable
## Rejected
Treat unavailable backend imports as evidence of implementation correctness or skip those backends
## Because
The audit must cover all advertised backends; environment availability determines reproduction strength, not scope
