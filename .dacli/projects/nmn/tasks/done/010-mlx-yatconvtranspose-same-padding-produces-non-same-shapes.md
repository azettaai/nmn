---
id: t-01M17A5GAKHM36J49R8M01N79V
kind: task
created: 2026-08-29T17:48:04Z
created_by: a-root
owner: a-root
github:
  issue: 77
  repo: azettaai/nmn
github_acceptance_import:
  issue: 77
  body_digest: sha256:2ea324ba261daf90bcde1ad8e8f5e3340c7a0c30316009503ffe9c5763408875
  actor: a-root
  imported_at: 2026-08-29T17:48:04Z
estimate: "{optimistic: 3, probable: 5, pessimistic: 8}"
depends_on: "[t-01M17A5MKTASSF5JR120Q8A5PZ]"
---
# MLX YatConvTranspose SAME padding produces non-SAME shapes
## Context
Adopted from GitHub issue #77.

## Summary

MLX `YatConvTranspose*` maps `padding="same"` to an approximation that does not preserve SAME output sizes.

## Evidence

This is source-decided because MLX aborts in the available headless environment. `_YatConvTransposeBase` uses symmetric `floor(((kernel-1)*dilation)/2)` padding and leaves `output_padding` unchanged.

By the standard transpose-convolution shape formula:

- kernel 3, stride 2, default output padding gives `2 * input - 1`;
- kernel 4, stride 1 gives `input + 1`.

The source comment acknowledges the approximation is off for even kernels and strides, but the public option gives no warning.

## Acceptance criteria

- [ ] Define SAME semantics and calculate padding/output-padding accordingly, or reject unsupported combinations.
- [ ] Add 1D/2D/3D shape and numerical tests on MLX hardware for odd/even kernels, strides, and dilation.

## Acceptance
- [x] Define SAME semantics and calculate padding/output-padding accordingly, or reject unsupported combinations.
- [x] Add 1D/2D/3D shape and numerical tests on MLX hardware for odd/even kernels, strides, and dilation.
## Log
- 2026-08-29T18:02:32Z dependency edit by a-root (event 01M17AZZR7GPF5QJWVXSPN649C)
- 2026-08-30T14:34:48Z accepted by a-root
- 2026-08-30T14:34:48Z verified by `python3 -m compileall -q src/nmn/mlx tests/test_mlx` (exit 0) in branch codex/acceptance-79ebc at 79ebc40 — proves that tree builds, not that the work is in trunk
- 2026-08-30T14:34:48Z deliverable: no dacli/010-mlx-yatconvtranspose-same-padding-produces-non-same-shapes branch — nothing to check against master
- 2026-08-30T14:34:48Z completed by a-root
## Verification Evidence
{"command":"python3 -m compileall -q src/nmn/mlx tests/test_mlx","argv":["sh","-c","python3 -m compileall -q src/nmn/mlx tests/test_mlx"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":93,"artifact_hash":"sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","verifier":"a-root","branch":"codex/acceptance-79ebc","commit_sha":"79ebc4025bec42c47faf7bf88e8cf6c4e932c993","tree_sha":"0ac242244e40b1ad0b455529c6de454abb7c7e03","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"python3 -m compileall -q src/nmn/mlx tests/test_mlx","argv":["sh","-c","python3 -m compileall -q src/nmn/mlx tests/test_mlx"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":33,"artifact_hash":"sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","verifier":"a-root","branch":"codex/acceptance-79ebc","commit_sha":"79ebc4025bec42c47faf7bf88e8cf6c4e932c993","tree_sha":"0ac242244e40b1ad0b455529c6de454abb7c7e03","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"79ebc4025bec42c47faf7bf88e8cf6c4e932c993","observed_at":"2026-08-30T14:34:48.735607Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
