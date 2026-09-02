---
id: t-01M17A5N54KX8WK3NGECV5R935
kind: task
created: 2026-08-29T17:48:09Z
created_by: a-root
owner: a-root
github:
  issue: 56
  repo: azettaai/nmn
github_acceptance_import:
  issue: 56
  body_digest: sha256:0e36a102bed09cc67494f22efa7f4d3e5d7d615514f7a7660eed3223ae8105b7
  actor: a-root
  imported_at: 2026-08-29T17:48:09Z
estimate: "{optimistic: 2, probable: 3, pessimistic: 5}"
---
# Keras YatEmbed and MultiHeadYatAttention are not registered for deserialization
## Context
Adopted from GitHub issue #56.

## Summary

Keras `YatEmbed` and `MultiHeadYatAttention` implement `get_config()` but are not registered as serializable Keras objects. Direct config deserialization and full-model loading therefore require user-supplied `custom_objects`, unlike `YatNMN` and the convolution classes.

## Affected code

- `src/nmn/keras/embed.py`
- `src/nmn/keras/attention.py`

## Reproduction

On commit `4f0298d`, Keras 3.14.1 with the JAX backend:

```python
import keras
from nmn.keras import YatEmbed, MultiHeadYatAttention

for layer in [YatEmbed(10, 4), MultiHeadYatAttention(4, 2)]:
    config = keras.saving.serialize_keras_object(layer)
    keras.saving.deserialize_keras_object(config)
```

Both raise `TypeError: Could not locate class ...`. The same round trip succeeds for `YatNMN` and `YatConv1D`.

## Impact

`.keras` models, cloned models, and serialized configs containing advertised embedding or attention layers cannot be restored normally.

## Acceptance criteria

- [ ] Register both classes under stable NMN package/name identifiers.
- [ ] Object-config serialization/deserialization works without `custom_objects`.
- [ ] `keras.models.clone_model` preserves layer configuration.
- [ ] Full `.keras` save/load preserves weights, outputs, dtype policy, and constant/learnable options.

## Acceptance
- [x] Register both classes under stable NMN package/name identifiers.
- [x] Object-config serialization/deserialization works without `custom_objects`.
- [x] `keras.models.clone_model` preserves layer configuration.
- [x] Full `.keras` save/load preserves weights, outputs, dtype policy, and constant/learnable options.
## Log
- 2026-08-30T11:39:01Z accepted by a-root
- 2026-08-30T11:39:01Z verified by `KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py` (exit 0) in branch codex/acceptance-2a209 at 2a209b9 — proves that tree builds, not that the work is in trunk
- 2026-08-30T11:39:01Z deliverable: no dacli/031-keras-yatembed-and-multiheadyatattention-are-not-registered-for-deserialization branch — nothing to check against master
- 2026-08-30T11:39:01Z completed by a-root
## Verification Evidence
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":31146,"artifact_hash":"sha256:b775d33093592022d0ab67d8b47c273f4ccc93c117296b2a416a604d7b50b60b","verifier":"a-root","branch":"codex/acceptance-2a209","commit_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","tree_sha":"eaff49d53d9a200fbf47f760a58c36d9a739e76a","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py","argv":["sh","-c","KERAS_BACKEND=jax PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_keras/test_issue_regressions.py tests/test_keras/test_low_precision_reductions.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":29128,"artifact_hash":"sha256:41655241b90e4be3df21fb22f6b5049ef68c1d170474817a08bb28b8ce865f6d","verifier":"a-root","branch":"codex/acceptance-2a209","commit_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","tree_sha":"eaff49d53d9a200fbf47f760a58c36d9a739e76a","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","observed_at":"2026-08-30T11:39:01.967124Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
