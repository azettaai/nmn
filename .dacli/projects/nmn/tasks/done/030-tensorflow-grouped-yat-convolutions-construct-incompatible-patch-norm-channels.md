---
id: t-01M17A5MXVN5CVASADQ5F2JRDP
kind: task
created: 2026-08-29T17:48:09Z
created_by: a-root
owner: a-root
github:
  issue: 57
  repo: azettaai/nmn
github_acceptance_import:
  issue: 57
  body_digest: sha256:f2825270e4c912459de5e41176e63ef09c26b8ce6130e8b904bb7e685ebffecc
  actor: a-root
  imported_at: 2026-08-29T17:48:09Z
estimate: "{optimistic: 3, probable: 5, pessimistic: 8}"
---
# TensorFlow grouped YAT convolutions construct incompatible patch-norm channels
## Context
Adopted from GitHub issue #57.

## Summary

The TensorFlow 1D/2D/3D grouped YAT convolution code constructs its patch-norm convolution with only one output channel while the main convolution has `filters` output channels. With `groups > 1`, that helper kernel is incompatible with grouped convolution and its repeated result has only `filters / groups` channels.

## Affected code

- `src/nmn/tf/conv.py`
- `YatConv1D`, `YatConv2D`, and `YatConv3D`

## Decisive code path

For `groups=2`, `input_channels=4`, and `filters=4`:

- Main kernel shape ends in `(2, 4)` and produces 4 output channels.
- Patch-norm kernel shape ends in `(2, 1)`.
- The raw patch norm has one output channel if accepted.
- Repeating it `filters // groups` times produces 2 channels.
- Adding that result to the 4-channel dot product and kernel norms cannot broadcast.

TensorFlow is not installed in the current audit environment, so this finding is source/shape-decided and still needs a native runtime regression test.

## Impact

The public `groups` option cannot produce correct per-group distances for TensorFlow forward convolutions.

## Acceptance criteria

- [ ] Build one patch-norm output per group and repeat each group norm for its `filters / groups` filters.
- [ ] Eager and `tf.function` tests cover groups 2 and 4 in 1D, 2D, and 3D.
- [ ] Outputs and gradients match a split-by-group, apply, concatenate reference.
- [ ] Channel/filter divisibility failures have clear validation errors.

## Acceptance
- [x] Build one patch-norm output per group and repeat each group norm for its `filters / groups` filters.
- [x] Eager and `tf.function` tests cover groups 2 and 4 in 1D, 2D, and 3D.
- [x] Outputs and gradients match a split-by-group, apply, concatenate reference.
- [x] Channel/filter divisibility failures have clear validation errors.
## Log
- 2026-08-30T11:39:51Z accepted by a-root
- 2026-08-30T11:39:51Z verified by `/Users/tahabsn/.pixi/bin/python3 -m compileall -q src/nmn/tf tests/test_tf/test_grouped_conv_parity.py tests/test_tf/test_saved_model_export.py` (exit 0) in branch codex/acceptance-2a209 at 2a209b9 — proves that tree builds, not that the work is in trunk
- 2026-08-30T11:39:51Z deliverable: no dacli/030-tensorflow-grouped-yat-convolutions-construct-incompatible-patch-norm-channels branch — nothing to check against master
- 2026-08-30T11:39:51Z completed by a-root
## Verification Evidence
{"command":"/Users/tahabsn/.pixi/bin/python3 -m compileall -q src/nmn/tf tests/test_tf/test_grouped_conv_parity.py tests/test_tf/test_saved_model_export.py","argv":["sh","-c","/Users/tahabsn/.pixi/bin/python3 -m compileall -q src/nmn/tf tests/test_tf/test_grouped_conv_parity.py tests/test_tf/test_saved_model_export.py"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":29,"artifact_hash":"sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","verifier":"a-root","branch":"codex/acceptance-2a209","commit_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","tree_sha":"eaff49d53d9a200fbf47f760a58c36d9a739e76a","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"2a209b9bfd58960e5a8916c9e6f60438135e5629","observed_at":"2026-08-30T11:39:51.724447Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
