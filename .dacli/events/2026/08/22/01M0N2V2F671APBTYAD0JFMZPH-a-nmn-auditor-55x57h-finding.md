---
id: 01M0N2V2F671APBTYAD0JFMZPH
kind: event
schema_version: 1
event_kind: finding
created: 2026-08-22T15:53:42Z
created_by: a-nmn-auditor-55x57h
about: "[[t-01M0N1M6AFM02BZN9S5FKKNQEN]]"
origin: agent
applied: true
checksum: sha256:788f593cc26d9db6da7a20d5a02bbab7675de87829d9edc44f26172798fe190f
---
Keras YatEmbed and MultiHeadYatAttention cannot deserialize without custom objects

Evidence: src/nmn/keras/embed.py:20 and src/nmn/keras/attention.py:175 define custom Layer classes with get_config but neither is exported/registered; only YatNMN and conv classes use keras_export. Keras 3.14.1 JAX repro: cfg=keras.saving.serialize_keras_object(YatEmbed(10,4)); keras.saving.deserialize_keras_object(cfg) raises TypeError Could not locate class YatEmbed; the same failure occurs for MultiHeadYatAttention(4,2), while YatNMN and YatConv1D round-trip. Impact: full-model .keras cloning/loading containing advertised embedding or attention fails unless users supply custom_objects. Tests only save/load YatNMN weights and do not deserialize these classes. Dedup: full GitHub issue list unavailable and targeted public search found no match. Acceptance: register both classes under a stable nmn package/name; object config, clone_model, and .keras save/load round-trip without custom_objects and preserve outputs/weights/config.
