---
id: f-pytorch-tied-dense-layers-reinitialize-an-existing-shared-kernel-bank
kind: note
note_kind: finding
created: 2026-08-22T21:11:13Z
created_by: a-root
about: "[[t-01M0N7AN7XAXWW4EN9R4DCFCRW]]"
severity: major
---
# PyTorch tied dense layers reinitialize an existing shared kernel bank
Runtime-reproduced: after creating a tied YatNMN and saving its weight/output, constructing a second compatible tied YatNMN shares the same Parameter but mutates it (max weight change 1.54; first-layer output change 0.64). __init__ assigns the existing shared Parameter and then unconditionally calls reset_parameters(), reinitializing the bank. This affects ordinary tied layers without lazy mode.
