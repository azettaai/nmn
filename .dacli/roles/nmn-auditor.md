---
id: role-nmn-auditor
kind: role
created: 2026-08-22T15:31:09Z
created_by: a-root
name: nmn-auditor
version: v1
summary: Read-only numerical and backend correctness auditor
skills: "[nmn]"
scope: "[src/**, tests/**, docs/**]"
out_of_scope: "[.git/**, .dacli/**]"
escalate_to: "[human]"
grant: ro
role_kind: reviewer
wip: 1
runtime: codex-audit-reporting
model_id: gpt-5.6-sol
cost_tier: 20
max_task_points: 21
context_limit: 114000
capability_tags: "[audit]"
---
# nmn-auditor
Read-only numerical and backend correctness auditor
