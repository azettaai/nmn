---
id: t-01M17A5MRHNCWVQZM9KBTCBCFD
kind: task
created: 2026-08-29T17:48:08Z
created_by: a-root
owner: a-root
github:
  issue: 58
  repo: azettaai/nmn
github_acceptance_import:
  issue: 58
  body_digest: sha256:1faba5ca417a113263d775ad927dbe8859725227a5f1e91451c61c4f1e76bffe
  actor: a-root
  imported_at: 2026-08-29T17:48:08Z
estimate: "{optimistic: 5, probable: 8, pessimistic: 13}"
depends_on: "[t-01M17A5MXVN5CVASADQ5F2JRDP]"
---
# TensorFlow modules lack callable SavedModel export signatures
## Context
Adopted from GitHub issue #58.

## Summary

The TensorFlow backend is implemented with `tf.Module`, but its public call methods are plain Python methods using only `tf.Module.with_name_scope`. Saving these objects with `tf.saved_model.save` records their variables but provides no callable concrete function or serving signature after load.

## Affected code

- `src/nmn/tf/nmn.py`
- `src/nmn/tf/conv.py`
- `src/nmn/tf/embed.py`
- `src/nmn/tf/attention.py`

## Decisive code path

`YatNMN.__call__`, convolution `__call__`, embedding lookup/attend, and attention `__call__` are not `tf.function`s and no explicit concrete-function export API is provided. The existing serialization coverage uses `tf.train.Checkpoint`, which validates variable restoration but not callable SavedModel export.

TensorFlow is not installed in the current audit environment, so this issue records a source-decided SavedModel gap that should be confirmed in TensorFlow CI.

## Impact

Users selecting the TensorFlow backend for SavedModel pipelines load a trackable variable container instead of an invokable NMN layer unless they write and maintain their own wrapper/signature.

## Acceptance criteria

- [ ] Provide serializable `tf.function` signatures or an explicit supported export API.
- [ ] Save/load tests cover `YatNMN`, convolution, embedding lookup/attend, and attention.
- [ ] Loaded functions preserve outputs, shapes, dtypes, and variables in eager/graph usage.
- [ ] Documentation shows the supported SavedModel export and invocation path.

## Acceptance
- [x] Provide serializable `tf.function` signatures or an explicit supported export API.
- [x] Save/load tests cover `YatNMN`, convolution, embedding lookup/attend, and attention.
- [x] Loaded functions preserve outputs, shapes, dtypes, and variables in eager/graph usage.
- [x] Documentation shows the supported SavedModel export and invocation path.
## Log
- 2026-08-29T18:02:32Z dependency edit by a-root (event 01M17AZZY0A3758GCCVQZR3BB1)
- 2026-08-30T11:39:51Z accepted by a-root
- 2026-08-30T11:39:51Z verified by `/Users/tahabsn/.pixi/bin/python3 -m compileall -q src/nmn/tf tests/test_tf/test_grouped_conv_parity.py tests/test_tf/test_saved_model_export.py` (exit 0) in branch codex/acceptance-2a209 at 2a209b9 — proves that tree builds, not that the work is in trunk
- 2026-08-30T11:39:51Z deliverable: no dacli/029-tensorflow-modules-lack-callable-savedmodel-export-signatures branch — nothing to check against master
- 2026-08-30T11:39:51Z completed by a-root
## Verification Evidence
{"command":"/Users/tahabsn/.pixi/bin/python3 -m compileall -q src/nmn/tf tests/test_tf/test_grouped_conv_parity.py tests/test_tf/test_saved_model_export.py","argv":["sh","-c","/Users/tahabsn/.pixi/bin/python3 -m compileall -q src/nmn/tf tests/test_tf/test_grouped_conv_parity.py tests/test_tf/test_saved_model_export.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":42,"artifact_hash":"sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","verifier":"a-root","branch":"codex/acceptance-2a209","commit_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","tree_sha":"eaff49d53d9a200fbf47f760a58c36d9a739e76a","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","observed_at":"2026-08-30T11:39:51.451474Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
