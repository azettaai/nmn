---
id: t-01M1H57XPDR8GPGR30TCR0MFAN
kind: task
created: 2026-09-02T13:34:24Z
created_by: a-root
owner: a-root
github:
  issue: 121
  repo: azettaai/nmn
github_acceptance_import:
  issue: 121
  body_digest: sha256:0edf0366dc35b4fb4f7522929d34e42b276ebd650559961e9b1b9cd40987917f
  actor: a-root
  imported_at: 2026-09-02T13:34:24Z
estimate: "{optimistic: 2, probable: 3, pessimistic: 5}"
---
# Make coverage uploads observable and enforce coverage policy
## Context
Adopted from GitHub issue #121.

## Problem

All three Codecov upload steps are displayed as successful even though the uploader reports `Token required - not valid tokenless upload`. The workflow uses `continue-on-error: true`, Codecov defaults `fail_ci_if_error` to false, the public badge is `unknown`, and there is no enforced coverage floor.

Evidence: https://github.com/azettaai/nmn/actions/runs/33630184902/job/100247427629

## Acceptance criteria

- [ ] Configure authenticated Codecov upload using OIDC or a valid repository secret.
- [ ] Set uploads to fail visibly when the coverage report cannot be uploaded.
- [ ] The public coverage badge reports a numeric value for `master`.
- [ ] Add a checked Codecov configuration with explicit project/patch policy or an equivalent local coverage floor.
- [ ] Add workflow-policy regression tests covering authentication and failure behavior.

## Non-goals

- Requiring identical coverage from optional backend jobs.
- Treating unavailable local backends as uncovered production regressions.

## Acceptance
- [ ] Configure authenticated Codecov upload using OIDC or a valid repository secret.
- [ ] Set uploads to fail visibly when the coverage report cannot be uploaded.
- [ ] The public coverage badge reports a numeric value for `master`.
- [ ] Add a checked Codecov configuration with explicit project/patch policy or an equivalent local coverage floor.
- [ ] Add workflow-policy regression tests covering authentication and failure behavior.
## Log
