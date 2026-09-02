---
id: 01M0N2NKQ5MC2JW2BSEHGRXJGZ
kind: event
schema_version: 1
event_kind: finding
created: 2026-08-22T15:50:43Z
created_by: a-nmn-auditor-55x57h
about: "[[t-01M0N1M6AFM02BZN9S5FKKNQEN]]"
origin: agent
applied: true
checksum: sha256:0e503a3fe8b72b02ff6e329a9fe9186641e6dd8a582d51d1df0513e7da2d5ab8
---
PyTorch low-precision conv and embedding can emit negative or infinite YAT scores

Evidence: src/nmn/torch/layers/_yat_conv_core.py:257-260 and src/nmn/torch/embed.py:135-144 form norm-sum minus 2 dot without clamping before division. CPU torch 2.11.0 repro with an exact kernel/query match: YatConv1D(1,1,3,bias=False,use_alpha=False,epsilon=1e-5,dtype=torch.float16), seed 0, x=weight.reshape(1,1,3) returns -228.125 because computed distance is -0.000244140625; bfloat16 seed 9 returns -2.984375. YatEmbed(1,3,use_alpha=False,epsilon=1e-5,dtype=float16), seed 0, query=embedding returns inf; bfloat16 seed 9 returns -97.5 with distance -0.015625. Impact: violates non-negative finite YAT semantics in advertised dtype behavior. Tests: float16 YatNMN test is CUDA-gated; conv/embed exact-match low-precision paths are absent; full torch suite passed. Dedup: full GitHub list unavailable and targeted public issue search found no match. Acceptance: clamp computed squared distances to at least zero for all conv/transpose-conv and both spherical/non-spherical embed paths; CPU float16/bfloat16 exact-match regressions must be finite and non-negative with finite gradients.
