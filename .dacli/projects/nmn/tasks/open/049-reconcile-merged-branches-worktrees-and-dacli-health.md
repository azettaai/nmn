---
id: t-01M1H57X8XSA5S1QJQFQJNXA0N
kind: task
created: 2026-09-02T13:34:24Z
created_by: a-root
owner: a-root
github:
  issue: 126
  repo: azettaai/nmn
github_acceptance_import:
  issue: 126
  body_digest: sha256:04e7006478a1137847a3ff7dfc71a451a82bab1f1fbeb25adc2989324fd29b9e
  actor: a-root
  imported_at: 2026-09-02T13:34:24Z
estimate: "{optimistic: 2, probable: 5, pessimistic: 8}"
---
# Reconcile merged branches, worktrees, and dacli health
## Context
Adopted from GitHub issue #126.

## Problem

The completed release workspace retains roughly 20 merged remote branches and many completed worktrees. GitHub has `delete_branch_on_merge=false`. Local dacli health is also degraded: role metadata is incomplete, the `nmn-auditor` read-only grant cannot be enforced by its runtime, ten events remain pending, and two detached acceptance worktrees cannot be classified safely by `dacli branches audit`.

## Acceptance criteria

- [ ] Enable automatic deletion of merged pull-request branches.
- [ ] Use `dacli branches audit` and its content-addressed safe-prune flow to remove only branches/worktrees proven merged and terminal.
- [ ] Repair role metadata required by `dacli doctor`.
- [ ] Give the read-only auditor an enforceable read-only runtime or correct its declared grant.
- [ ] Reconcile pending events and detached acceptance worktrees without deleting durable run evidence.
- [ ] `dacli doctor --json` reports healthy and branch audit has no unknown worktrees.

## Non-goals

- Deleting durable transcripts or evidence records.
- Manually deleting branches whose merge state is unknown.

## Acceptance
- [ ] Enable automatic deletion of merged pull-request branches.
- [ ] Use `dacli branches audit` and its content-addressed safe-prune flow to remove only branches/worktrees proven merged and terminal.
- [ ] Repair role metadata required by `dacli doctor`.
- [ ] Give the read-only auditor an enforceable read-only runtime or correct its declared grant.
- [ ] Reconcile pending events and detached acceptance worktrees without deleting durable run evidence.
- [ ] `dacli doctor --json` reports healthy and branch audit has no unknown worktrees.
## Log
