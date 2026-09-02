---
id: 01M0N7GYCVDMRW313J181D7CHP
kind: event
schema_version: 1
event_kind: finding
created: 2026-08-22T17:15:33Z
created_by: a-nmn-auditor-rys7wb
about: "[[t-01M0N7AN7XAXWW4EN9R4DCFCRW]]"
origin: agent
applied: true
checksum: sha256:f27d96843623c453e4e24101425ef1a56a8b3cca1b2edc5ad1009a800c5b5b9f
---
PyTorch float16 dense YatNMN overflows at exact collisions despite distance clamping

Repro on CPU torch 2.11.0 against the explicit direct-distance reference: YatNMN(3,1,bias=False,alpha=False,epsilon=1e-5,dtype=param_dtype=torch.float16), weight=[.25,-.5,.75], x=weight returns inf and autograd input gradient is non-finite in both spherical=False and spherical=True. src/nmn/torch/nmn/yat_nmn.py:343-374 clamps distance but performs numerator square, divide, and output in float16; the finite mathematical score is about 7.66e4, beyond float16 max. Impact: advertised low-precision dense training can silently produce inf/NaN at high-similarity inputs even after the cancellation clamp. tests/test_torch/test_all_layers.py:259-272 only checks a random CUDA float16 forward and dtype, with no collision or gradient assertion. Semantic dedup: distinct from sibling f-pytorch-low-precision-conv-and-embedding because this is the dense implementation and requires a dense compute/output policy (upcast, bounded result, or documented unsupported range), not only a non-negative distance clamp; GitHub inspection is unavailable because dacli reports project nmn is not linked and linking is a consent step. Acceptance: add CPU float16/bfloat16 exact/near-collision dense forward+gradient reference tests across spherical/ordinary modes and make the supported contract finite or explicitly constrain it.
