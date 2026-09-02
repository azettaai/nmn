---
id: t-01M1H57XS3WAHJ2Z0KZZGP7R3Z
kind: task
created: 2026-09-02T13:34:25Z
created_by: a-root
owner: a-root
github:
  issue: 120
  repo: azettaai/nmn
github_acceptance_import:
  issue: 120
  body_digest: sha256:031635c0f078c7ad978bab90e5cf2e3f9795d46f488e2f398e499230f3e631c0
  actor: a-root
  imported_at: 2026-09-02T13:34:25Z
estimate: "{optimistic: 2, probable: 3, pessimistic: 8}"
---
# Restore the broken public mirror and sync v0.3.4
## Context
Adopted from GitHub issue #120.

## Problem

The canonical repository advertises `mlnomadpy/nmn` as a synchronized public mirror, but every recent mirror workflow run fails because `MIRROR_PAT` is empty. The mirror is stuck at commit `4f0298d` from 2026-07-01 and does not contain the `v0.3.4` tag.

Evidence:

- Failed release run: https://github.com/azettaai/nmn/actions/runs/33631027239
- The log reports `MIRROR_PAT is not set; the public mirror was not updated.`
- Canonical `master`: `bea00ced976b953f4ce85f6820f6f01b9c46a94e`
- Mirror `master`: `4f0298d3bf61c2bb8266e07eb7599169a7977e55`

## Acceptance criteria

- [ ] Configure a least-privilege credential or documented replacement authentication mechanism that can write to `mlnomadpy/nmn`.
- [ ] A manual mirror workflow run completes successfully.
- [ ] Mirror `master` exactly matches canonical `master`.
- [ ] Mirror contains the annotated `v0.3.4` tag at the canonical release commit.
- [ ] The workflow continues to fail visibly when authentication or verification is unavailable.

## Non-goals

- Changing the canonical repository.
- Hiding mirror failures with `continue-on-error`.

## Acceptance
- [ ] Configure a least-privilege credential or documented replacement authentication mechanism that can write to `mlnomadpy/nmn`.
- [ ] A manual mirror workflow run completes successfully.
- [ ] Mirror `master` exactly matches canonical `master`.
- [ ] Mirror contains the annotated `v0.3.4` tag at the canonical release commit.
- [ ] The workflow continues to fail visibly when authentication or verification is unavailable.
## Log
