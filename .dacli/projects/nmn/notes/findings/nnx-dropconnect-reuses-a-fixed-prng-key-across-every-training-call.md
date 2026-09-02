---
id: f-nnx-dropconnect-reuses-a-fixed-prng-key-across-every-training-call
kind: note
note_kind: finding
created: 2026-08-22T21:00:50Z
created_by: a-root
about: "[[t-01M0N7AN7XAXWW4EN9R4DCFCRW]]"
severity: major
---
# NNX DropConnect reuses a fixed PRNG key across every training call
Evidence: src/nmn/nnx/layers/nmn.py lines 346-347 stores one rngs.params key and lines 385-387 calls jax.random.bernoulli with that unchanged key on every forward. On JAX 0.9.2, two consecutive deterministic=False calls to YatNMN(64,64,use_dropconnect=True,drop_rate=0.5,rngs=nnx.Rngs(0)) with identical input are exactly equal with max difference 0. The same stored-key pattern appears in NNX YatConv and YatConvTranspose. Expected: training calls consume or receive fresh randomness; deterministic=True remains repeatable. Tests only check shape/finiteness and do not check mask evolution. GitHub search found no semantic duplicate.
