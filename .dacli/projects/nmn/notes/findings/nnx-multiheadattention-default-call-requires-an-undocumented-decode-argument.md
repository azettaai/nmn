---
id: f-nnx-multiheadattention-default-call-requires-an-undocumented-decode-argument
kind: note
note_kind: finding
created: 2026-08-22T21:45:19Z
created_by: a-root
about: "[[t-01M0N7ATD4SKWBNH770NRGBY2Y]]"
severity: moderate
---
# NNX MultiHeadAttention default call requires an undocumented decode argument
Runtime-reproduced using the class docstring's basic call pattern: constructing with default decode=None and calling m(x, deterministic=True) raises ValueError saying no decode argument was provided. Tests consistently pass decode=False, while the public example and ordinary non-autoregressive usage do not. The default should resolve to False rather than making every call specify it.
