---
id: t-01M17A5JQ0AR0H1ZSGEZ841P62
kind: task
created: 2026-08-29T17:48:06Z
created_by: a-root
owner: a-root
github:
  issue: 68
  repo: azettaai/nmn
github_acceptance_import:
  issue: 68
  body_digest: sha256:27c7692f80863e326fb83a62a3f7445d28553ce2f4f02b8d5563ac45ddbb0068
  actor: a-root
  imported_at: 2026-08-29T17:48:06Z
estimate: "{optimistic: 3, probable: 5, pessimistic: 8}"
depends_on: "[t-01M17A5KA5PQ0SP0KYRM1S3HBR]"
---
# NNX MultiHeadAttention ignores out_features and omits output projection
## Context
Adopted from GitHub issue #68.

## Summary

NNX `MultiHeadAttention` stores `out_features`, `out_kernel_init`, and `out_bias_init` but never creates or applies an output projection.

## Reproduction

```python
m = MultiHeadAttention(
    num_heads=2, in_features=8, qkv_features=8,
    out_features=3, decode=False, rngs=nnx.Rngs(0),
)
m(jnp.ones((1, 4, 8)), deterministic=True).shape
# (1, 4, 8), expected (1, 4, 3)
```

Existing tests set `out_features == qkv_features`, hiding the defect. The documented basic call also fails unless callers explicitly pass `decode=False`, despite decode being optional in the example.

## Acceptance criteria

- [ ] Create/apply the output projection and honor its initializer/bias options.
- [ ] Test differing input, QKV, and output dimensions.
- [ ] Make ordinary non-decoding calls default to `decode=False` or correct the API/docs consistently.

## Acceptance
- [x] Create/apply the output projection and honor its initializer/bias options.
- [x] Test differing input, QKV, and output dimensions.
- [x] Make ordinary non-decoding calls default to `decode=False` or correct the API/docs consistently.
## Log
- 2026-08-29T18:02:30Z dependency edit by a-root (event 01M17AZY9RRYS5X1SPMKVMF3YX)
- 2026-08-30T09:11:27Z accepted by a-root
- 2026-08-30T09:11:27Z verified by `PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'honors_output_projection'` (exit 0) in branch codex/acceptance-b88 at b88e5ed — proves that tree builds, not that the work is in trunk
- 2026-08-30T09:11:27Z deliverable: no dacli/019-nnx-multiheadattention-ignores-out-features-and-omits-output-projection branch — nothing to check against master
- 2026-08-30T09:11:27Z completed by a-root
## Verification Evidence
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'honors_output_projection'","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'honors_output_projection'"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":3525,"artifact_hash":"sha256:ceda6e798ff0c1dd8d4a0f4d44804d51f85a01c9b6aef453303cc20fae42bfca","verifier":"a-root","branch":"codex/acceptance-b88","commit_sha":"b88e5eda007cb2baa1205d9e803d4aff81665aad","tree_sha":"fb080c6983893af1e0e3e4bc5eacb87a42d2d970","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"}}
{"command":"PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'honors_output_projection'","argv":["sh","-c","PYTHONPATH=src /Users/tahabsn/.pixi/bin/python3 -m pytest -q tests/test_nnx/test_attention_regressions.py -k 'honors_output_projection'"],"working_directory":"/Users/tahabsn/Documents/GitHub/nmn","exit_code":0,"duration_ms":3266,"artifact_hash":"sha256:329e386af57c3e3d6e1a025b93ba8c62952fc3ad05ca8f7977a227ffe7143858","verifier":"a-root","branch":"codex/acceptance-b88","commit_sha":"b88e5eda007cb2baa1205d9e803d4aff81665aad","tree_sha":"fb080c6983893af1e0e3e4bc5eacb87a42d2d970","clean":true,"runtime_versions":{"arch":"arm64","go":"go1.26.5","os":"darwin"},"tool_versions":{"git":"git version 2.50.1 (Apple Git-155)","shell":"/bin/sh (3.2.57(1)-release)"},"external":[{"provider":"github","head_sha":"b88e5eda007cb2baa1205d9e803d4aff81665aad","observed_at":"2026-08-30T09:11:27.591024Z","state":"unobservable","skip_reason":"observe exact-commit GitHub checks: check your internet connection or https://githubstatus.com (exit 1; next: inspect the retained stderr/stdout tail and correct the command condition)"}]}
