---
id: 01M0NMXQZQCSXGH5Y8VGHVXE5T
kind: event
schema_version: 1
event_kind: finding
created: 2026-08-22T21:09:44Z
created_by: a-nmn-auditor-qqf66n
about: "[[t-01M0N7AN7XAXWW4EN9R4DCFCRW]]"
origin: agent
applied: true
checksum: sha256:ecdcefa52b43f64e9c8c9c80afc63448f25c7a18234976ee0b4d0a3dedd259a8
---
PyTorch lazy tied kernel globally freezes eager peer layers

Pixi/PyTorch 2.11 repro after clearing YatNMN._KERNEL_BANKS: create eager=YatNMN(3,2,tie_kernel_bank=True,kernel_bank_id="audit",lazy=False), then frozen=YatNMN(...same bank...,lazy=True). eager.weight is frozen.weight is True, eager.weight.requires_grad changes from True to False, and backward through eager leaves eager.weight.grad=None. Creating the lazy layer first and an eager peer second also leaves both false. src/nmn/torch/nmn/yat_nmn.py:135-179 stores and reuses one nn.Parameter for the bank, then lines 250-255 calls requires_grad_(False) on that shared object for a lazy instance. NNX explicitly rejects lazy plus tie_kernel_bank at HEAD:170-174, demonstrating the unsafe combination is recognized elsewhere. Impact: adding a frozen consumer silently stops training all existing and future eager consumers of a shared bank; order does not rescue it. rg finds no PyTorch test combining tie_kernel_bank with lazy, while the NNX rejection is tested at tests/test_nnx/test_yatnmn_lazy.py:54-56. Acceptance: reject the combination in PyTorch or implement per-consumer stop-gradient without mutating shared Parameter trainability, and test both construction orders plus optimizer/gradient behavior. Open local tasks contain no semantic duplicate; GitHub issue dedup remains unavailable because api.github.com cannot be reached.
