---
id: 01M0NMYFKV1ZVNMQ5W7QRDAWGX
kind: event
schema_version: 1
event_kind: finding
created: 2026-08-22T21:10:08Z
created_by: a-nmn-auditor-qqf66n
about: "[[t-01M0N7AN7XAXWW4EN9R4DCFCRW]]"
origin: agent
applied: true
checksum: sha256:f22e36d994257fc57e7d3510cb008ad7f7515ffe795bd85c73336a138245c72d
---
PyTorch tied-bank construction reinitializes existing layers

Seeded PyTorch 2.11 repro: clear YatNMN._KERNEL_BANKS, create a=YatNMN(3,2,tie_kernel_bank=True,kernel_bank_id="reset-audit",bias=False,alpha=False), snapshot its weight/output, then create b with the same configuration. a.weight is b.weight, but construction of b changes a weight (max abs change 1.6122448) and its fixed-input output (max abs change 4.4381418). src/nmn/torch/nmn/yat_nmn.py:138-167 correctly reuses the existing Parameter, but unconditional reset_parameters at line 224 invokes kernel_init on self.weight at lines 237-250 even when that Parameter came from the bank. Impact: model assembly order silently destroys already initialized, loaded, or trained shared directions and changes earlier layer behavior. rg finds no PyTorch tied-bank tests. Acceptance: initialize only newly created/expanded bank entries, never reinitialize an existing shared Parameter, and test that adding same-size/smaller/larger consumers preserves all pre-existing slices and outputs. This is distinct from the lazy tied-bank finding: it occurs with two eager consumers. Local open tasks contain no duplicate; GitHub semantic dedup remains unverified because api.github.com is unreachable.
