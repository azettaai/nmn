# Executable documentation examples

These small examples are executed by the backend CI jobs. Install the relevant
optional framework first; constructors intentionally differ between backends.
See the [API references](typing.md) for the full signatures.

<!-- backend: torch -->
```python
import torch
from nmn.torch import YatNMN
layer = YatNMN(in_features=8, out_features=4)
assert layer(torch.ones(2, 8)).shape == (2, 4)
```

<!-- backend: nnx -->
```python
import jax.numpy as jnp
from flax import nnx
from nmn.nnx import YatNMN
layer = YatNMN(in_features=8, out_features=4, rngs=nnx.Rngs(0))
assert layer(jnp.ones((2, 8))).shape == (2, 4)
```

<!-- backend: tf -->
```python
import tensorflow as tf
from nmn.tf import YatNMN
layer = YatNMN(features=4)
assert tuple(layer(tf.ones((2, 8))).shape) == (2, 4)
```

The CLI commands work even without an optional framework:

```sh
python -m nmn info
python -m nmn frameworks
python -m nmn guide torch
python -m nmn features
```
