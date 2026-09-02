---
id: f-nnx-freeze-kernel-false-silently-disables-lazy-true-unlike-every-other-backend
kind: note
note_kind: finding
created: 2026-08-22T21:13:27Z
created_by: a-nmn-auditor-qqf66n
about: "[[t-01M0N7AN7XAXWW4EN9R4DCFCRW]]"
source_event: 01M0NMT9B7TT5CQ5GJP1MG6K1E
---
# NNX freeze_kernel false silently disables lazy true unlike every other backend
Reproduction with JAX 0.9.2: YatNMN(3,1,lazy=True,freeze_kernel=False,rngs=nnx.Rngs(0)) resolves model.lazy=False and nnx.state(model,nnx.Param) contains kernel; lazy=True with freeze_kernel omitted excludes kernel, and freeze_kernel=True also excludes it. src/nmn/nnx/layers/nmn.py HEAD:161-176 assigns lazy=bool(freeze_kernel) whenever the alias is non-None. Torch uses bool(lazy or freeze_kernel), Linen checks lazy or freeze_kernel, Keras uses bool(lazy or freeze_kernel), TF uses bool(lazy or freeze_kernel), and MLX does likewise, matching the repository contract that if either is true the kernel is frozen. Impact: portable configuration plumbing that supplies a default false alias can unintentionally train NNX feature directions while all other backends freeze them. tests/test_nnx/test_yatnmn_lazy.py:48-52 explicitly asserts the divergent override, so the gap is semantic rather than absent execution. Acceptance: resolve with logical OR, update the contradictory test, and add a six-backend truth-table regression for both flags plus optimizer/trainable-state checks for bias, alpha, epsilon, and kernel. Open local task listing has no duplicate; GitHub issue inspection failed because api.github.com is unreachable, so remote semantic dedup remains unverified.
