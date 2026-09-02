---
id: t-01M17A5MKTASSF5JR120Q8A5PZ
kind: task
created: 2026-08-29T17:48:08Z
created_by: a-root
owner: a-root
github:
  issue: 59
  repo: azettaai/nmn
github_acceptance_import:
  issue: 59
  body_digest: sha256:09c36ca9bb911301285af983e8bdb1e7d71e9b01a8d2fdb66bd6e2bd107354d4
  actor: a-root
  imported_at: 2026-08-29T17:48:08Z
estimate: "{optimistic: 2, probable: 4, pessimistic: 7}"
---
# MLX fused YatNMN detaches learnable epsilon from autodiff
## Context
Adopted from GitHub issue #59.

## Summary

The MLX `YatNMN(fused=True, learnable_epsilon=True)` path converts the softplus-constrained epsilon array to a Python float before calling the fused custom-VJP function. That removes `epsilon_param` from the autodiff graph or can fail during tracing.

## Affected code

- `src/nmn/mlx/nmn.py`
- `src/nmn/mlx/fused.py`

## Decisive code path

The module computes:

```python
eps_val = float(nn.softplus(self.epsilon_param)[0])
```

and `fused_yat_score` then reconstructs a new MLX array from that Python scalar. Although the fused core computes an internal epsilon gradient, it cannot flow back through the detached float conversion to `epsilon_param`.

MLX aborts during Metal initialization in this audit environment, so this is source-decided and requires confirmation on a working Apple GPU runtime.

## Impact

Fused mode violates the advertised rule that learnable epsilon remains trainable, including in lazy/frozen-kernel training.

## Acceptance criteria

- [ ] Pass epsilon as an MLX array through `fused_yat_score` without Python scalar conversion.
- [ ] The custom VJP propagates epsilon gradients back to `epsilon_param` through softplus.
- [ ] Fused and eager outputs and kernel/bias/alpha/epsilon gradients match.
- [ ] Tests cover `mx.value_and_grad`, `mx.compile`, and lazy mode on a supported MLX runtime.

## Acceptance
- [x] Pass epsilon as an MLX array through `fused_yat_score` without Python scalar conversion.
- [x] The custom VJP propagates epsilon gradients back to `epsilon_param` through softplus.
- [x] Fused and eager outputs and kernel/bias/alpha/epsilon gradients match.
- [x] Tests cover `mx.value_and_grad`, `mx.compile`, and lazy mode on a supported MLX runtime.
## Log
- 2026-08-30T14:35:00Z accepted by a-root
- 2026-08-30T14:35:00Z verified by `python3 -m compileall -q src/nmn/mlx tests/test_mlx` (exit 0) in branch codex/acceptance-79ebc at 79ebc40 — proves that tree builds, not that the work is in trunk
- 2026-08-30T14:35:00Z deliverable: no dacli/028-mlx-fused-yatnmn-detaches-learnable-epsilon-from-autodiff branch — nothing to check against master
- 2026-08-30T14:35:00Z completed by a-root
## Verification Evidence
{"command":"python3 -m compileall -q src/nmn/mlx tests/test_mlx","argv":["sh","-c","python3 -m compileall -q src/nmn/mlx tests/test_mlx"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":93,"artifact_hash":"sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","verifier":"a-root","branch":"codex/acceptance-79ebc","commit_sha":"79ebc4025bec42c47faf7bf88e8cf6c4e932c993","tree_sha":"0ac242244e40b1ad0b455529c6de454abb7c7e03","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"python3 -m compileall -q src/nmn/mlx tests/test_mlx","argv":["sh","-c","python3 -m compileall -q src/nmn/mlx tests/test_mlx"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":40,"artifact_hash":"sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","verifier":"a-root","branch":"codex/acceptance-79ebc","commit_sha":"79ebc4025bec42c47faf7bf88e8cf6c4e932c993","tree_sha":"0ac242244e40b1ad0b455529c6de454abb7c7e03","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"79ebc4025bec42c47faf7bf88e8cf6c4e932c993","observed_at":"2026-08-30T14:35:00.703017Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
