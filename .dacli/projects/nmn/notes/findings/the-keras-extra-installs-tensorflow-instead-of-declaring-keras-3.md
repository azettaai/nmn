---
id: f-the-keras-extra-installs-tensorflow-instead-of-declaring-keras-3
kind: note
note_kind: finding
created: 2026-08-22T21:52:10Z
created_by: a-root
about: "[[t-01M0N7B0WY37YQY22JWDA1B4GC]]"
severity: moderate
---
# The Keras extra installs TensorFlow instead of declaring Keras 3
Packaging/source finding: pyproject.toml defines keras = ['tensorflow>=2.10.0'] even though nmn.keras is documented as Keras 3 multi-backend and runs successfully with KERAS_BACKEND=jax without TensorFlow. This forces a large unrelated backend, prevents a clean JAX/PyTorch Keras install, and does not express the actual keras>=3 API dependency. The tf extra already exists separately.
