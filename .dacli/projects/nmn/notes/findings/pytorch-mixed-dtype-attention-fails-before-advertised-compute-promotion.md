---
id: f-pytorch-mixed-dtype-attention-fails-before-advertised-compute-promotion
kind: note
note_kind: finding
created: 2026-08-22T15:55:04Z
created_by: a-nmn-auditor-55x57h
about: "[[t-01M0N1M6AFM02BZN9S5FKKNQEN]]"
source_event: 01M0N2NKPGGQBAKV0CAPHDH1C4
---
# PyTorch mixed dtype attention fails before advertised compute promotion
Evidence: src/nmn/torch/attention/multi_head.py:106-119 stores projection weights in param_dtype, but forward projects raw inputs at lines 193-196 and only promotes q/k/v afterward at lines 218-219. Minimal repro on torch 2.11.0: MultiHeadYatAttention(4,2,dtype=torch.float32,param_dtype=torch.float64)(torch.randn(1,2,4,dtype=torch.float32)) raises RuntimeError: mat1 and mat2 must have the same dtype, but got Float and Double. Impact: documented dtype/param_dtype separation is unusable for attention and mixed-precision calls fail. Tests: tests/test_torch has dtype checks for YatNMN/conv but no attention mixed-storage execution; full torch suite passed. Dedup: dacli GitHub link unavailable and targeted public issue search surfaced no semantic match. Acceptance: a regression using unequal compute/storage dtypes completes forward/backward, returns compute dtype, preserves parameter storage dtype, and passes state_dict round-trip.
