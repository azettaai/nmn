---
id: t-01M1H57XH7HECKBSHHV6VNS5YT
kind: task
created: 2026-09-02T13:34:24Z
created_by: a-root
owner: a-root
github:
  issue: 123
  repo: azettaai/nmn
github_acceptance_import:
  issue: 123
  body_digest: sha256:4a87df9eb498140d23bba653bcd7056f58961e57fe22aaee3ff32e110eb3910d
  actor: a-root
  imported_at: 2026-09-02T13:34:24Z
estimate: "{optimistic: 3, probable: 5, pessimistic: 8}"
---
# Close CI trigger gaps and harden workflow supply chain
## Context
Adopted from GitHub issue #123.

## Problem

CI policy tests only trigger for `src/**`, `tests/**`, `pyproject.toml`, and `test.yml`. Changes to publication, mirroring, deployment, package manifests, security/release docs, or developer commands can bypass the tests that validate those surfaces.

Additional hardening gaps:

- third-party actions use mutable tags and repository Actions allow every action with no SHA-pinning requirement;
- publish workflow grants `id-token: write` to the build job;
- publish/mirror/deploy jobs have no explicit timeouts;
- minimum dependency versions are continuously tested only for JAX;
- PyTorch is CPU-only and native TPU validation is not continuous.

## Acceptance criteria

- [ ] Trigger policy/quality CI for every workflow and packaging/developer-policy file it validates.
- [ ] Add explicit job timeouts to publish, mirror, and deployment workflows.
- [ ] Scope OIDC permission only to jobs that publish or deploy.
- [ ] Pin third-party Actions to immutable commits, with update automation retained.
- [ ] Document and test a sustainable minimum-version policy for Torch, TensorFlow, Keras, and MLX, or raise unsupported lower bounds.
- [ ] Document continuous accelerator coverage and clearly identify TPU/CUDA gaps.
- [ ] Add workflow-policy tests for the new invariants.

## Non-goals

- Duplicating every backend suite on every Python and device combination.
- Adding long-lived cloud credentials to pull-request workflows.

## Acceptance
- [ ] Trigger policy/quality CI for every workflow and packaging/developer-policy file it validates.
- [ ] Add explicit job timeouts to publish, mirror, and deployment workflows.
- [ ] Scope OIDC permission only to jobs that publish or deploy.
- [ ] Pin third-party Actions to immutable commits, with update automation retained.
- [ ] Document and test a sustainable minimum-version policy for Torch, TensorFlow, Keras, and MLX, or raise unsupported lower bounds.
- [ ] Document continuous accelerator coverage and clearly identify TPU/CUDA gaps.
- [ ] Add workflow-policy tests for the new invariants.
## Log
