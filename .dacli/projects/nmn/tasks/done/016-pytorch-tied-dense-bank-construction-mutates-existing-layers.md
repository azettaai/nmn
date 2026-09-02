---
id: t-01M17A5HPD5E247R54ZS01JPCA
kind: task
created: 2026-08-29T17:48:05Z
created_by: a-root
owner: a-root
github:
  issue: 71
  repo: azettaai/nmn
github_acceptance_import:
  issue: 71
  body_digest: sha256:681baf86ffda4e84ad1afbd4db9801aaec4561d3e0114a61e2007ae394c051af
  actor: a-root
  imported_at: 2026-08-29T17:48:05Z
estimate: "{optimistic: 2, probable: 4, pessimistic: 7}"
---
# PyTorch tied dense bank construction mutates existing layers
## Context
Adopted from GitHub issue #71.

## Summary

Constructing a later PyTorch tied dense `YatNMN` mutates already-live peer layers that share the bank.

## Reproduction

Creating a second compatible tied layer unconditionally calls `reset_parameters()` on the already-shared `Parameter`. A seeded reproduction changed existing weights by `1.54` and the first layer's output by `0.64`.

Separately, constructing a tied `lazy=True` layer calls `requires_grad_(False)` on the shared `Parameter`, globally freezing eager peer layers.

## Impact

Model construction order changes existing model outputs and trainability.

## Acceptance criteria

- [ ] Initialize a bank only when it is first created.
- [ ] Do not mutate shared parameter-wide trainability from one consumer's local lazy option; reject incompatible sharing or use per-consumer gradient handling.
- [ ] Test output and trainability invariance while adding tied consumers.

## Acceptance
- [x] Initialize a bank only when it is first created.
- [x] Do not mutate shared parameter-wide trainability from one consumer's local lazy option; reject incompatible sharing or use per-consumer gradient handling.
- [x] Test output and trainability invariance while adding tied consumers.
## Log
- 2026-08-30T10:47:26Z accepted by a-root
- 2026-08-30T10:47:26Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py` (exit 0) in branch codex/acceptance-dc696 at dc69677 — proves that tree builds, not that the work is in trunk
- 2026-08-30T10:47:26Z deliverable: dacli/016-pytorch-tied-dense-bank-construction-mutates-existing-layers exists but is NOT in master — closed anyway
- 2026-08-30T10:47:26Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":2255,"artifact_hash":"sha256:8db14daa5fcc79e612025435b3ea76ad662ba636e03f0a9030b092c611da801c","verifier":"a-root","branch":"codex/acceptance-dc696","commit_sha":"dc69677f09cc2b37ffc7bced3fddb71265cc061f","tree_sha":"9ba1e98ad5e7fbb58b7b080c3fb54c607919e2cd","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_issue_regressions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":2066,"artifact_hash":"sha256:bd16be27936932b3f7d1450931ed68e86a6143bfe52af7fb02f68bdace10bc9a","verifier":"a-root","branch":"codex/acceptance-dc696","commit_sha":"dc69677f09cc2b37ffc7bced3fddb71265cc061f","tree_sha":"9ba1e98ad5e7fbb58b7b080c3fb54c607919e2cd","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"dc69677f09cc2b37ffc7bced3fddb71265cc061f","observed_at":"2026-08-30T10:47:26.28541Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
