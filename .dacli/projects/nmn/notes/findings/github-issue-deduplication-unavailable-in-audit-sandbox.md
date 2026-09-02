---
id: f-github-issue-deduplication-unavailable-in-audit-sandbox
kind: note
note_kind: finding
created: 2026-08-22T15:44:04Z
created_by: a-nmn-auditor-187qrm
about: "[[t-01M0N1M68973FWMQFG54QNSSCV]]"
source_event: 01M0N22W335AFM7QKEG6YZ8ANX
---
# GitHub issue deduplication unavailable in audit sandbox
dacli github sync nmn --dry-run reports project nmn is not linked; direct gh issue list --repo azettaai/nmn failed connecting to api.github.com. Existing GitHub issues therefore cannot be inspected from this run, and any defect recommendation must remain unfiled pending owner-side issue deduplication.
