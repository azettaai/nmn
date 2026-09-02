---
id: t-01M1H57XEN9J4GQY81P6M5H3S1
kind: task
created: 2026-09-02T13:34:24Z
created_by: a-root
owner: a-root
github:
  issue: 124
  repo: azettaai/nmn
github_acceptance_import:
  issue: 124
  body_digest: sha256:dcb0319330cb571f2cc87cd30d94cfc17cf08d3de18250df997e1cba0c7a87c1
  actor: a-root
  imported_at: 2026-09-02T13:34:24Z
estimate: "{optimistic: 1, probable: 2, pessimistic: 3}"
---
# Repair developer workflow, documentation, and template drift
## Context
Adopted from GitHub issue #124.

## Problem

Developer commands, documentation, and contribution templates drifted after the CI/type-check overhaul:

- `make build` fails after documented `make install` because the `build` package is absent from dev dependencies;
- `make typecheck` checks only Torch and NNX while CI checks all `src/nmn`;
- plain `python -m pytest` can import a stale site-package instead of the checkout because `src/` is not placed first;
- `SECURITY.md` lists 0.2.x rather than current 0.3.x;
- `CONTRIBUTING.md` incorrectly says `nmn[keras]` installs TensorFlow and describes MyPy/format checks as advisory;
- declared backend pytest markers are unused although docs recommend them;
- MLX is absent from PR and issue framework selectors;
- pre-commit tool versions drift from CI.

## Acceptance criteria

- [ ] `make install && make build` works in a clean environment.
- [ ] `make typecheck` invokes the same package-wide MyPy configuration as CI.
- [ ] Pytest reliably imports `src/nmn` when run from a checkout, with a regression test against stale installed-package shadowing.
- [ ] Security and contribution documentation matches v0.3.x, optional extras, and blocking CI behavior.
- [ ] Either apply the documented backend/integration markers consistently or remove the unused marker-based instructions/configuration.
- [ ] Add MLX to contribution and issue templates.
- [ ] Align pre-commit tool versions with CI or centralize the versions.

## Non-goals

- Installing every framework for a single-backend contribution.

## Acceptance
- [x] `make install && make build` works in a clean environment.
- [x] `make typecheck` invokes the same package-wide MyPy configuration as CI.
- [x] Pytest reliably imports `src/nmn` when run from a checkout, with a regression test against stale installed-package shadowing.
- [x] Security and contribution documentation matches v0.3.x, optional extras, and blocking CI behavior.
- [x] Either apply the documented backend/integration markers consistently or remove the unused marker-based instructions/configuration.
- [x] Add MLX to contribution and issue templates.
- [x] Align pre-commit tool versions with CI or centralize the versions.
## Log
- 2026-09-02T13:45:05Z completion requested by a-root; PR landing state no branch on master
- 2026-09-02T14:01:24Z accepted by a-root
- 2026-09-02T14:01:24Z verified by `python3 -m pytest tests/test_workflow_policy.py -q` (exit 0) in branch master at 3e957d1 — proves that tree builds, not that the work is in trunk
- 2026-09-02T14:01:24Z deliverable: dacli/051-repair-developer-workflow-documentation-and-template-drift is merged into master
- 2026-09-02T14:01:24Z completed by a-root
- 2026-09-02T14:01:26Z deliverable: dacli/051-repair-developer-workflow-documentation-and-template-drift is merged into master
## Verification Evidence
{"command":"python3 -m pytest tests/test_workflow_policy.py -q","argv":["sh","-c","python3 -m pytest tests/test_workflow_policy.py -q"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":546,"artifact_hash":"sha256:afb1b3407df2ed9e5211393eec797affc9197d05916b8508cf3540bd14f62742","verifier":"a-root","branch":"codex/124-developer-workflow","commit_sha":"e610ee159e0f019fc7338fe56b6178a17cc0863a","tree_sha":"0545e7988853afb796c8fb4a3f5807ef70049508","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.22.12","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"python3 -m pytest tests/test_workflow_policy.py -q","argv":["sh","-c","python3 -m pytest tests/test_workflow_policy.py -q"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":415,"artifact_hash":"sha256:07e2c252b164780cae8dcf671fd6e1d38cae7237905c8a385d751ff4ef975fd2","verifier":"a-root","branch":"master","commit_sha":"3e957d16344560a3575d8b9044c2f0f796796734","tree_sha":"0545e7988853afb796c8fb4a3f5807ef70049508","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.22.12","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github-actions","workflow_run_id":"33639336166","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639336166","name":"Code Quality: Push on master","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-actions","workflow_run_id":"33639337079","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337079","name":"Mirror to mlnomadpy/nmn","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-actions","workflow_run_id":"33639337141","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337141","name":"Test Suite","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278143674","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639336166/job/100278143674","name":"Analyze (javascript-typescript)","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278144228","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639336166/job/100278144228","name":"Analyze (python)","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278140064","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337141/job/100278140064","name":"Lint","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278138276","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337079/job/100278138276","name":"Push master + tags to mlnomadpy/nmn","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278140078","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337141/job/100278140078","name":"Test JAX/Flax (latest, py3.12)","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278139866","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337141/job/100278139866","name":"Test JAX/Flax (minimum, py3.11)","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278139638","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337141/job/100278139638","name":"Test Keras (jax backend)","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278139775","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337141/job/100278139775","name":"Test Keras (torch backend)","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278139735","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337141/job/100278139735","name":"Test Keras/TF (macos-latest, py3.11)","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278140027","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337141/job/100278140027","name":"Test Keras/TF (ubuntu-latest, py3.10)","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278139998","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337141/job/100278139998","name":"Test Keras/TF (ubuntu-latest, py3.11)","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278139732","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337141/job/100278139732","name":"Test Keras/TF (ubuntu-latest, py3.12)","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278139711","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337141/job/100278139711","name":"Test MLX (Apple Silicon)","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278139860","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337141/job/100278139860","name":"Test PyTorch (macos-latest, py3.11)","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278139914","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337141/job/100278139914","name":"Test PyTorch (ubuntu-latest, py3.10)","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278139920","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337141/job/100278139920","name":"Test PyTorch (ubuntu-latest, py3.11)","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278139742","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337141/job/100278139742","name":"Test PyTorch (ubuntu-latest, py3.12)","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278140378","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337141/job/100278140378","name":"Test PyTorch (windows-latest, py3.11)","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278139490","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337141/job/100278139490","name":"Test base install and optional-backend collection","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"},{"provider":"github-check","check_run_id":"100278139755","head_sha":"3e957d16344560a3575d8b9044c2f0f796796734","url":"https://github.com/azettaai/nmn/actions/runs/33639337141/job/100278139755","name":"Type Check (MyPy, package surface)","observed_at":"2026-09-02T14:01:16.801558Z","state":"pending"}]}
