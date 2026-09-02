---
id: t-01M198FZN6ZA0YTEGY3AEK9W9S
kind: task
created: 2026-08-30T11:57:19Z
created_by: a-root
owner: a-root
github:
  issue: 94
  repo: azettaai/nmn
github_acceptance_import:
  issue: 94
  body_digest: sha256:b4849e22f528f3ed8b481f05bdcbc0505aee0c31737c5f97b325ed06e2ec3b6b
  actor: a-root
  imported_at: 2026-08-30T11:57:19Z
---
# Learnable epsilon initialization is unstable for valid extreme values across backends
## Context
Adopted from GitHub issue #94.

## Summary

Torch, Linen, Keras, and TensorFlow initialize the learnable-epsilon raw parameter with log(exp(epsilon)-1). This underflows for small positive epsilon and overflows for large epsilon even though the public API accepts positive values. NNX already uses a stable inverse-softplus.

## Reproduction

Constructing YatNMN with epsilon=1e-20 and learnable_epsilon=True raises in Torch, Linen, Keras, and TensorFlow; the same expression appears in their convolution families.

## Acceptance criteria

- [ ] Use a stable inverse-softplus such as epsilon + log(-expm1(-epsilon)) across affected dense and convolution families.
- [ ] Validate epsilon is finite and strictly positive before parameter creation.
- [ ] Add tiny, ordinary, and large epsilon construction/forward/gradient tests across affected backends.
- [ ] Preserve existing constructor defaults and serialization.

## Acceptance
- [x] Use a stable inverse-softplus such as epsilon + log(-expm1(-epsilon)) across affected dense and convolution families.
- [x] Validate epsilon is finite and strictly positive before parameter creation.
- [x] Add tiny, ordinary, and large epsilon construction/forward/gradient tests across affected backends.
- [x] Preserve existing constructor defaults and serialization.
## Log
- 2026-08-30T13:37:21Z accepted by a-root
- 2026-08-30T13:37:21Z verified by `KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_stable_learnable_epsilon.py tests/test_linen/test_stable_learnable_epsilon.py tests/test_keras/test_stable_learnable_epsilon.py` (exit 0) in branch codex/acceptance-7a09 at 7a09b10 — proves that tree builds, not that the work is in trunk
- 2026-08-30T13:37:21Z deliverable: no dacli/045-learnable-epsilon-initialization-is-unstable-for-valid-extreme-values-across branch — nothing to check against master
- 2026-08-30T13:37:21Z completed by a-root
## Verification Evidence
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_stable_learnable_epsilon.py tests/test_linen/test_stable_learnable_epsilon.py tests/test_keras/test_stable_learnable_epsilon.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_stable_learnable_epsilon.py tests/test_linen/test_stable_learnable_epsilon.py tests/test_keras/test_stable_learnable_epsilon.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":18759,"artifact_hash":"sha256:5af582a430dd18772c43107fbd19ce8972033d54d017dc5b8c008de4dfff1421","verifier":"a-root","branch":"codex/acceptance-7a09","commit_sha":"7a09b1093a3745db10424eae95f9139b328078ed","tree_sha":"ae31324ea0d781460d6fdb72309bafec767394c2","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_stable_learnable_epsilon.py tests/test_linen/test_stable_learnable_epsilon.py tests/test_keras/test_stable_learnable_epsilon.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_torch/test_stable_learnable_epsilon.py tests/test_linen/test_stable_learnable_epsilon.py tests/test_keras/test_stable_learnable_epsilon.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":18746,"artifact_hash":"sha256:c579049ee0e6dda27b54c1da1c9200f3861b635e4f3798a1671b4e942eec6b81","verifier":"a-root","branch":"codex/acceptance-7a09","commit_sha":"7a09b1093a3745db10424eae95f9139b328078ed","tree_sha":"ae31324ea0d781460d6fdb72309bafec767394c2","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"7a09b1093a3745db10424eae95f9139b328078ed","observed_at":"2026-08-30T13:37:21.163042Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
