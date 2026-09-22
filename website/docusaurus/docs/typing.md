# Typing support

NMN ships `py.typed`. Public functions and constructors expose annotated scalar
options, optional arguments and backend-native tensor results. Framework
tracing, shapes, lazy parameter creation, and serialization are runtime contracts;
the type checker does not prove numerical compatibility or shape correctness.

Keras uses a documented `Tensor = Any` boundary because its runtime backend may
produce TensorFlow, JAX, Torch, or symbolic Keras tensors. Initializers,
regularizers, constraints, extensible serialized configs, and forwarded Keras
layer kwargs use `Any` because Keras accepts registered objects and callables.
Other backends retain their native tensor types. This is gradual typing, not a
claim of strict typing for every private framework implementation.

CI checks every public export's signature and docstring from the source export
lists, checks the package with MyPy, and runs a consumer against an installed
wheel. Strict definition checking is enabled per module and expanded as private
implementation annotations mature. Generated API pages are refreshed with
`python scripts/generate_api_docs.py` and verified with `--check`.

- [PyTorch API](api/torch.md)
- [Flax NNX API](api/nnx.md)
- [Flax Linen API](api/linen.md)
- [TensorFlow API](api/tf.md)
- [Keras API](api/keras.md)
- [MLX API](api/mlx.md)
