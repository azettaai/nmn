---
id: f-keras-float16-and-bfloat16-conv-embed-emit-infinite-or-negative-yat-scores
kind: note
note_kind: finding
created: 2026-08-22T15:55:04Z
created_by: a-nmn-auditor-55x57h
about: "[[t-01M0N1M6AFM02BZN9S5FKKNQEN]]"
source_event: 01M0N2V2EP49MGCM4REWGVPEER
---
# Keras float16 and bfloat16 conv/embed emit infinite or negative YAT scores
Evidence: src/nmn/keras/conv.py:347-348, 736-737, 1095-1096, 1432-1433, 1725-1726, 2025-2026 and src/nmn/keras/embed.py:126-135 compute squared distance without a non-negative clamp. Keras 3.14.1 with JAX 0.9.2 backend repro: exact query/kernel match, dimension/kernel size 3, random key 0 in float16 makes both YatEmbed.attend and YatConv1D return inf; key 8 in bfloat16 computes distance -0.015625 and both return -69. Impact: silent violation of finite non-negative YAT semantics in supported dtype policies across all six conv variants and embedding attend. Tests: Keras suite reported 41 passed and 71 skipped under JAX; no low-precision exact-match checks. Dedup: full GitHub list unavailable and targeted public search found no match. Acceptance: clamp distances to zero before epsilon for embed and all conv/transpose variants; JAX-backed float16/bfloat16 exact-match forward and gradient regressions are finite/non-negative, with equivalent TF-backend coverage.
