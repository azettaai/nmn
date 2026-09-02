---
id: 01M0N2NKV00PKPKB768B4C975X
kind: event
schema_version: 1
event_kind: finding
created: 2026-08-22T15:50:43Z
created_by: a-nmn-auditor-55x57h
about: "[[t-01M0N1M6AFM02BZN9S5FKKNQEN]]"
origin: agent
applied: true
checksum: sha256:f8828daf558fe58a5b1a17ae12bbe7f1e37e9eb2722a06070dd98cb555d25d7f
---
Linen grouped YatConv1D 2D and 3D always fail XLA validation

Evidence: src/nmn/linen/conv.py:116-127, 245-256, and 375-386 build the patch-norm ones kernel with exactly one output feature while passing feature_group_count. XLA requires RHS output features divisible by the group count. CPU JAX 0.9.2 repro: YatConv1D(features=4,kernel_size=(3,),feature_group_count=2).init(key,ones((1,5,4))) raises ValueError: rhs output feature dimension size must be a multiple of feature_group_count, but 1 is not a multiple of 2; identical failures reproduce for 2D and 3D. Impact: an advertised constructor option is unusable for all forward conv dimensions. Tests: rg finds no Linen feature_group_count regression; all 133 Linen tests passed. Dedup: full GitHub issue list unavailable and targeted public search found no match. Acceptance: grouped 1D/2D/3D init/apply work for at least groups 2 and 4, match a per-group reference, validate channel/filter divisibility clearly, and preserve jit/gradient behavior.
