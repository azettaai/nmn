---
id: 01M0N2V2FPSYDA8HYDMCVQDX7A
kind: event
schema_version: 1
event_kind: finding
created: 2026-08-22T15:53:42Z
created_by: a-nmn-auditor-55x57h
about: "[[t-01M0N1M6AFM02BZN9S5FKKNQEN]]"
origin: agent
applied: true
checksum: sha256:3ae210bd261c9fb573e587adb14fde8bb29b2051fe70386cf0482c58961b9c96
---
TensorFlow grouped forward convolutions have incompatible patch-norm channel construction

Decisive code path: src/nmn/tf/conv.py:99-101 builds grouped dot kernels with input_channels/groups and filters outputs, but patch-norm kernels at 173-186, 367-380, and 562-575 have one total output channel and then repeat it only filters/groups times. For groups=2, the resulting patch tensor has filters/2 channels while dot_prod_map and kernel norms have filters channels at lines 197, 391, and 586, so addition cannot broadcast; grouped conv kernels may also be rejected because one output channel is not divisible by groups. Impact: groups>1 cannot compute correct per-group distances in 1D/2D/3D. TensorFlow is absent locally, so evidence is source/shape-decisive; tests contain no groups case. Dedup: full GitHub issue list unavailable and targeted public search found no match. Acceptance: construct one patch-norm output per group, repeat each filters/groups times, and add TF eager plus tf.function tests for groups 2/4 against a split-concat reference with gradients.
