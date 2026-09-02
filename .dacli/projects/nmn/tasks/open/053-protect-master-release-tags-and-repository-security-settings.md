---
id: t-01M1H57XKXSY3G8ZYJATHBP32S
kind: task
created: 2026-09-02T13:34:24Z
created_by: a-root
owner: a-root
github:
  issue: 122
  repo: azettaai/nmn
github_acceptance_import:
  issue: 122
  body_digest: sha256:fc7b40da11c79582861f441c7022e28f29de13c5dfaec4e0714bb0edbd05066b
  actor: a-root
  imported_at: 2026-09-02T13:34:24Z
estimate: "{optimistic: 2, probable: 3, pessimistic: 8}"
---
# Protect master, release tags, and repository security settings
## Context
Adopted from GitHub issue #122.

## Problem

The default branch has no branch protection, the only repository ruleset is disabled, releases are mutable, secret scanning and push protection are disabled, and Dependabot security updates are disabled. A version tag can trigger trusted publication without a repository rule requiring the tested default branch.

The PyPI environment has a required reviewer, which mitigates publication risk but does not protect source integration or tags.

## Acceptance criteria

- [ ] Enable an enforced default-branch ruleset requiring pull requests and the repository's required test/type/lint checks.
- [ ] Prevent force pushes and branch deletion on `master`.
- [ ] Protect release tags matching `v*.*.*` or enforce an equivalent release rule.
- [ ] Enable secret scanning, push protection, and Dependabot security updates where supported.
- [ ] Enable automatic deletion of merged branches.
- [ ] Document which repository settings are required for release integrity.

## Non-goals

- Removing the existing PyPI environment approval.
- Granting GitHub Actions write permissions broadly.

## Acceptance
- [ ] Enable an enforced default-branch ruleset requiring pull requests and the repository's required test/type/lint checks.
- [ ] Prevent force pushes and branch deletion on `master`.
- [ ] Protect release tags matching `v*.*.*` or enforce an equivalent release rule.
- [ ] Enable secret scanning, push protection, and Dependabot security updates where supported.
- [ ] Enable automatic deletion of merged branches.
- [ ] Document which repository settings are required for release integrity.
## Log
