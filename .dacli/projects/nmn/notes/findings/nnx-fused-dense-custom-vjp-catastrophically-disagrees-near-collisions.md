---
id: f-nnx-fused-dense-custom-vjp-catastrophically-disagrees-near-collisions
kind: note
note_kind: finding
created: 2026-08-22T21:13:27Z
created_by: a-nmn-auditor-qqf66n
about: "[[t-01M0N7AN7XAXWW4EN9R4DCFCRW]]"
source_event: 01M0NMWQGWKDFKVFD57SHCZF93
---
# NNX fused dense custom VJP catastrophically disagrees near collisions
Deterministic pixi/JAX 0.9.2 repro: width=64, w=jax.random.normal(jax.random.key(164),(64,1))*100, x=w[:,0]*(1.00001), bias=1, alpha disabled, epsilon=1e-3. YatNMN fused=False and fused=True have identical forward output, while float32 cancellation makes the optimized raw distance -0.125 before clamp. Their input gradients differ catastrophically (maximum relative error about 4,988x in repeated reproduction); other exact/near-collision cases yield a fused zero gradient where the standard gradient is finite and nonzero. In released HEAD src/nmn/nnx/layers/nmn.py:489-504 clamps raw distance with maximum, but the custom backward at 553-591 always differentiates the unclamped distance and assembles large g_dot/g_dist terms that catastrophically cancel; it neither saves the clamp activity nor uses autodiff of the exact forward graph. Impact: fused dense training silently follows a different gradient precisely for high-similarity samples where YAT scores are largest. tests/test_nnx/test_fused_yatnmn.py:179-268 compares only independent random inputs and has no exact/near-collision or negative-raw-distance case. The owner worktree currently refactors this code but retains the optimized default custom VJP, so the defect remains reproducible without modifying user changes. Acceptance: fused forward and all input/kernel/bias/alpha/epsilon gradients match a direct-distance explicit oracle and standard autodiff at exact and near collisions, including negative rounded raw distances, ranks 1-3, jit, float32 and supported low precision; mutation of clamp handling must make the regression fail. Open local tasks contain no duplicate; GitHub semantic dedup is unverified because api.github.com is unreachable.
