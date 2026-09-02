---
id: f-pytorch-tied-yatconv-kernels-never-receive-gradients
kind: note
note_kind: finding
created: 2026-08-22T21:40:36Z
created_by: a-root
about: "[[t-01M0N7AQVHA9TCHD11Z9CDT9RH]]"
severity: major
---
# PyTorch tied YatConv kernels never receive gradients
Runtime-reproduced for YatConv1D/2D/3D with tie_kernel_bank=True: after sum(output).backward(), the restored shared weight.grad is None while alpha/bias gradients exist. forward temporarily replaces self.weight with nn.Parameter(original_weight[slice]), so autograd accumulates into the temporary leaf and the finally block discards it. This occurs even when the bank size exactly equals out_channels.
