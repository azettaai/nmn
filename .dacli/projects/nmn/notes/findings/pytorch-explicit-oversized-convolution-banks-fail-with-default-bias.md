---
id: f-pytorch-explicit-oversized-convolution-banks-fail-with-default-bias
kind: note
note_kind: finding
created: 2026-08-22T21:40:36Z
created_by: a-root
about: "[[t-01M0N7AQVHA9TCHD11Z9CDT9RH]]"
severity: major
---
# PyTorch explicit oversized convolution banks fail with default bias
Runtime-reproduced: YatConv1D(in=2,out=3,kernel=3,tie_kernel_bank=True,kernel_bank_size=5) fails in _yat_score because dot output has 3 channels but parent Conv allocated a 5-element bias. The same constructor pattern exists in 2D/3D. bias=False bypasses this particular crash but exposes the separate lost-gradient defect.
