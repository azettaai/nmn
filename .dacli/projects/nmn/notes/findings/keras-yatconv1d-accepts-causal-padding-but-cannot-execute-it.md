---
id: f-keras-yatconv1d-accepts-causal-padding-but-cannot-execute-it
kind: note
note_kind: finding
created: 2026-08-22T21:16:29Z
created_by: a-root
about: "[[t-01M0N7AQVHA9TCHD11Z9CDT9RH]]"
severity: moderate
---
# Keras YatConv1D accepts causal padding but cannot execute it
Runtime-reproduced with the Keras JAX backend: YatConv1D(filters=3,kernel_size=3,padding='causal') builds but call raises RuntimeError: Unrecognized padding type, expected VALID/SAME/SAME_LOWER, got causal. The implementation bypasses the base Conv1D causal-padding helper and passes 'causal' directly to keras.ops.conv for both dot and patch norm. valid and same execute successfully.
