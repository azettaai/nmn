---
id: t-01M1FJJKB7R5RA3M08000DMWKB
kind: task
created: 2026-09-01T22:48:57Z
created_by: a-root
owner: a-root
priority: should
estimate: "{optimistic: 1, probable: 2, pessimistic: 4}"
---
# Complete MyPy coverage with zero package exclusions
## So that
new backend modules and examples cannot silently escape static analysis
## Acceptance
- [x] mypy checks every src/nmn Python module with no exclude configuration
- [x] typing fixes preserve backend runtime behavior and targeted tests pass
- [x] CI policy prevents future package exclusions and the full non-slow suite passes
## Log
- 2026-09-01T22:49:08Z claimed by a-root
- 2026-09-01T23:18:10Z completion requested by a-root; PR landing state no branch on master
- 2026-09-01T23:18:20Z accepted by a-root
- 2026-09-01T23:18:20Z verified by `/private/tmp/nmn-mypy-ci/bin/mypy --no-error-summary && PYTHONPATH=src python -m pytest tests/test_workflow_policy.py -q` (exit 0) in branch master at 13bbaca — proves that tree builds, not that the work is in trunk
- 2026-09-01T23:18:20Z deliverable: no dacli/047-complete-mypy-coverage-with-zero-package-exclusions branch — nothing to check against master
- 2026-09-01T23:18:20Z completed by a-root
## Verification Evidence
{"command":"/private/tmp/nmn-mypy-231/bin/mypy --no-error-summary \u0026\u0026 PYTHONPATH=src python -m pytest tests/test_workflow_policy.py -q","argv":["sh","-c","/private/tmp/nmn-mypy-231/bin/mypy --no-error-summary \u0026\u0026 PYTHONPATH=src python -m pytest tests/test_workflow_policy.py -q"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":548,"artifact_hash":"sha256:0d106734540676ac241836dd0ff0092149caca704ebb094848e49d7ac472c5d1","verifier":"a-root","branch":"codex/114-complete-mypy-coverage","commit_sha":"21a2f1ea34b4701e8e5aed601fe31c5de5fac2c4","tree_sha":"620e71a39fe6cbe2158e5f6428a8faa2ab78e605","runtime_versions":{"arch":"arm64","go":"go1.22.12","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"/private/tmp/nmn-mypy-ci/bin/mypy --no-error-summary \u0026\u0026 PYTHONPATH=src python -m pytest tests/test_workflow_policy.py -q","argv":["sh","-c","/private/tmp/nmn-mypy-ci/bin/mypy --no-error-summary \u0026\u0026 PYTHONPATH=src python -m pytest tests/test_workflow_policy.py -q"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":1218,"artifact_hash":"sha256:44ed97700f218b18bd10a26d7298fc90489cee7958fb11c509865a4cc835ca6b","verifier":"a-root","branch":"master","commit_sha":"13bbacaa589f321fb841b40eae087352855f537d","tree_sha":"b42054dff0ca1b546de1d122df7e3ab7c1dd7868","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.22.12","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"13bbacaa589f321fb841b40eae087352855f537d","observed_at":"2026-09-01T23:18:20.727416Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
