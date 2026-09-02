---
id: 01M0N2V2G3HTACJGT6REM5WM5K
kind: event
schema_version: 1
event_kind: finding
created: 2026-08-22T15:53:42Z
created_by: a-nmn-auditor-55x57h
about: "[[t-01M0N1M6AFM02BZN9S5FKKNQEN]]"
origin: agent
applied: true
checksum: sha256:482de6f7230c53975a7886fe2ce3ebaf71b4026531317ff09a14857ff3951782
---
TensorFlow modules do not survive SavedModel as callable layers

Decisive code path: advertised TensorFlow classes are tf.Module objects, but YatNMN.__call__ at src/nmn/tf/nmn.py:192-203, YatEmbed methods at src/nmn/tf/embed.py:88-101, attention __call__ at src/nmn/tf/attention.py:327-358, and conv calls such as src/nmn/tf/conv.py:147-158 are plain Python methods decorated only with tf.Module.with_name_scope, not tf.function/concrete signatures. tf.saved_model.save serializes variables but cannot restore these Python call methods or a serving signature. Impact: the backend advertised for TF/SavedModel pipelines loads as a trackable variable container rather than an invokable layer unless the user writes a wrapper. Existing serialization test tests/test_tf/test_comprehensive.py:83-114 exercises tf.train.Checkpoint only, not SavedModel. TensorFlow is absent locally; this follows TensorFlow SavedModel trace requirements. Dedup: full GitHub list unavailable and targeted public search found no match. Acceptance: provide serializable tf.function signatures or an explicit export API for implemented modules; save/load YatNMN, conv, embed attend/lookup, and attention must expose callable concrete functions and preserve outputs in eager and graph mode.
