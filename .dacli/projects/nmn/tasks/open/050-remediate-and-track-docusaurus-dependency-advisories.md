---
id: t-01M1H57XC0TTB3F08EFQD2B0N9
kind: task
created: 2026-09-02T13:34:24Z
created_by: a-root
owner: a-root
github:
  issue: 125
  repo: azettaai/nmn
github_acceptance_import:
  issue: 125
  body_digest: sha256:199a94fe598ce89ebe2b23efe19806461bd01f08f6cf08f12003a5ec15615e92
  actor: a-root
  imported_at: 2026-09-02T13:34:24Z
estimate: "{optimistic: 2, probable: 3, pessimistic: 8}"
---
# Remediate and track Docusaurus dependency advisories
## Context
Adopted from GitHub issue #125.

## Problem

`npm audit --omit=dev --audit-level=high` reports 24 advisories in the locked Docusaurus dependency graph: 18 high and 6 moderate. High findings include `image-size` parser denial-of-service advisories and `serialize-javascript` RCE/CPU-exhaustion advisories. npm currently reports no direct fix for the high-severity transitive paths.

Relevant advisories:

- https://github.com/advisories/GHSA-w3rx-r6r6-pgpr
- https://github.com/advisories/GHSA-5p2g-fcmc-qvqq
- https://github.com/advisories/GHSA-5c6j-r48x-rmvq
- https://github.com/advisories/GHSA-qj8w-gfj5-8c6v

## Acceptance criteria

- [ ] Apply every non-breaking lockfile remediation currently available, including `uuid`.
- [ ] Determine whether patched Docusaurus/transitive releases or safe overrides resolve each high advisory.
- [ ] Document residual no-fix advisories and why repository-only trusted build inputs limit exposure.
- [ ] Add a CI audit policy that fails on newly introduced high/critical advisories without making accepted no-fix advisories permanently red.
- [ ] Re-run the Docusaurus production build after dependency changes.

## Non-goals

- Suppressing advisories without an explicit reviewed allowlist.
- Replacing Docusaurus solely to make the audit output empty.

## Acceptance
- [ ] Apply every non-breaking lockfile remediation currently available, including `uuid`.
- [ ] Determine whether patched Docusaurus/transitive releases or safe overrides resolve each high advisory.
- [ ] Document residual no-fix advisories and why repository-only trusted build inputs limit exposure.
- [ ] Add a CI audit policy that fails on newly introduced high/critical advisories without making accepted no-fix advisories permanently red.
- [ ] Re-run the Docusaurus production build after dependency changes.
## Log
