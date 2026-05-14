# MLX Guide

End-to-end walkthrough for using NMN with **MLX** — Apple's array
framework for ML on Apple Silicon.

- **Module path**: `nmn.mlx`
- **Minimum versions**: Python 3.10+, [`mlx`](https://pypi.org/project/mlx/) ≥ 0.18
- **Hardware**: Apple Silicon (`darwin-arm64`). MLX has no Linux/Windows
  wheels at the time of writing.

---

## 1. Install

```bash
pip install "nmn[mlx]"
```

Verify:

```python
import mlx.core as mx
import nmn
from nmn.mlx import YatNMN

print(nmn.__version__, mx.default_device())
layer = YatNMN(features=64)
y = layer(mx.zeros((32, 128)))
print(y.shape)  # (32, 64)
```

MLX automatically uses the integrated Metal GPU on Apple Silicon. To run
on CPU instead set the default device explicitly:

```python
import mlx.core as mx
mx.set_default_device(mx.cpu)
```

---

## 2. Hello world — replace one `Linear + ReLU`

```python
import mlx.core as mx
from nmn.mlx import YatNMN

# Before:
# fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU())

# After:
fc = YatNMN(features=64)

x = mx.random.normal(shape=(32, 128))
y = fc(x)
print(y.shape)            # (32, 64)
print(float(mx.min(y)))   # ≥ 0 (Yat output is a squared ratio)
```

No activation function — the non-linearity is baked into the layer.

---

## 3. MNIST end-to-end

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from nmn.mlx import YatNMN

class YatMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = YatNMN(features=256)
        self.fc2 = YatNMN(features=128)
        self.out = nn.Linear(128, 10)
        # Eagerly build the lazy YatNMN parameters so value_and_grad
        # captures the full parameter tree.
        dummy = mx.zeros((1, 28 * 28))
        _ = self.fc2(self.fc1(dummy))

    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        return self.out(self.fc2(self.fc1(x)))

def loss_fn(model, x, y):
    return mx.mean(nn.losses.cross_entropy(model(x), y))

model = YatMLP()
optimizer = optim.AdamW(learning_rate=3e-4)
grad_fn = nn.value_and_grad(model, loss_fn)

# ... your data loader → (x, y) batches ...
for x, y in batches:
    loss, grads = grad_fn(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
```

A complete runnable version is at
[`src/nmn/mlx/examples/mnist.py`](../../src/nmn/mlx/examples/mnist.py):

```bash
PYTHONPATH=src python -m nmn.mlx.examples.mnist \
    --epochs 3 --device gpu --report .context/mlx_mnist.json
```

**Measured baseline** (Apple Silicon, Metal GPU, fp32, batch=128, lr=3e-4, seed=0):

| Epoch | Train loss | Test loss | Test acc |
| ----- | ---------- | --------- | -------- |
| 0     | 0.4490     | 0.1923    | 93.89 %  |
| 1     | 0.1630     | 0.1448    | 95.56 %  |
| 2     | 0.1271     | 0.1220    | **96.28 %** |

3 epochs ≈ 1.6 s wall time. `--device cpu` reproduces the **same**
accuracies in ≈ 2.4 s. See the
[cross-framework benchmark blog post](../../website/blog-pages/20-cross-framework-mnist.html)
for a side-by-side with PyTorch / NNX / Linen.

---

## 4. CNN — replacing `Conv2d + ReLU`

```python
import mlx.core as mx
import mlx.nn as nn
from nmn.mlx import YatConv2D, YatNMN

class YatCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.c1 = YatConv2D(filters=32, kernel_size=3, padding="same")
        self.c2 = YatConv2D(filters=64, kernel_size=3, padding="same")
        self.fc = YatNMN(features=num_classes)
        # Warm up lazy params with a dummy CIFAR-shaped input.
        _ = self.__call__(mx.zeros((1, 32, 32, 3)))

    def __call__(self, x):
        x = nn.MaxPool2d(2)(self.c1(x))
        x = nn.MaxPool2d(2)(self.c2(x))
        x = mx.mean(x, axis=(1, 2))   # global average pool
        return self.fc(x)
```

Input layout is **channels-last** (`(N, H, W, C)`) — same as TF / Keras /
JAX. Supported `padding`: `"valid"`, `"same"`, or an integer / tuple of
integers for explicit symmetric padding.

---

## 5. Attention

```python
from nmn.mlx import MultiHeadYatAttention
import mlx.core as mx

attn = MultiHeadYatAttention(embed_dim=512, num_heads=8)
x = mx.random.normal(shape=(4, 16, 512))
y = attn(x)                # self-attention
y = attn(x, ctx, ctx)      # cross-attention with ctx as keys/values
```

For causal LMs, build a triangular boolean mask of shape
`(B, H, q_len, kv_len)` (`True = attend`) and pass it via `mask=`.

### Spherical-YAT-Performer — linear-complexity attention

For long sequences, the standard ``O(n²)`` softmax YAT attention is
replaced by an anchor-based feature decomposition that runs in
``O(n)``:

```python
from nmn.mlx import create_yat_tp_projection, yat_tp_attention
import mlx.core as mx

# Precompute the random projections once.
params = create_yat_tp_projection(
    head_dim=64,
    num_anchor_features=32,   # P
    num_prf_features=8,       # M
    num_quad_nodes=1,         # R
    seed=0,
)

q = mx.random.normal(shape=(2, 4096, 8, 64))  # long sequence
k = mx.random.normal(shape=(2, 4096, 8, 64))
v = mx.random.normal(shape=(2, 4096, 8, 64))
out = yat_tp_attention(q, k, v, params, causal=False)
```

Use ``causal=True`` for autoregressive decoding (prefix sums over the
key dimension). The total feature count is ``F = R·P·M``; memory
crosses below quadratic attention when ``seq_len > F``.

### Rotary Position Embeddings (RoPE)

```python
from nmn.mlx import RotaryYatAttention
import mlx.core as mx

attn = RotaryYatAttention(embed_dim=512, num_heads=8, max_seq_len=2048)
x = mx.random.normal(shape=(4, 16, 512))
y = attn(x)                                  # adds rotary positional encoding
y = attn(x, position_offset=128)             # caller-controlled offset
```

Composes with the standard YAT softmax over the same `(q', k')` rotation
pair. Free-form functional access through
`precompute_freqs_cis`, `apply_rotary_emb`,
`rotary_yat_attention_weights`, `rotary_yat_attention` for custom blocks.

---

## 6. Embeddings with weight tying

```python
from nmn.mlx import YatEmbed
import mlx.core as mx

embed = YatEmbed(num_embeddings=10_000, features=128, constant_alpha=True)

token_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
h = embed(token_ids)               # (1, 5, 128)
scores = embed.attend(h[0, -1])    # (10_000,) — YAT retrieval scoring
```

Same `.attend(query)` API as the other backends — useful for tying
input/output embeddings in language models.

---

## 7. Saving & loading

MLX serialises parameters to `.safetensors`:

```python
import mlx.utils

mlx.utils.tree_flatten(model.parameters())  # inspect
model.save_weights("yat_mlp.safetensors")

# Later:
restored = YatMLP()
restored.load_weights("yat_mlp.safetensors")
```

---

## 8. Lazy build pattern — read this once

`YatNMN`, `YatConv*`, `YatConvTranspose*`, and `MultiHeadYatAttention`
**build their parameters on the first forward pass** (matching the TF /
Keras backends' ergonomics). This means:

1. `nn.value_and_grad(model, loss_fn)` captures the parameter tree at
   *construction* time. If you wrap an unbuilt model, the captured
   tree won't include the projection kernels — gradients will only
   flow through the always-eager `alpha`.
2. Same for `optimizer.update(model, grads)` — the optimizer state
   shape is fixed at first call.

**Fix**: run one forward pass with a dummy input *before* wrapping the
model. The MNIST script above does this inside `__init__`; do the same
for your custom modules.

---

## 9. DropConnect regularization

`YatNMN`, `YatConv{1,2,3}D`, and `YatConvTranspose{1,2,3}D` all accept a
weight-level DropConnect:

```python
layer = YatNMN(features=128, use_dropconnect=True, drop_rate=0.2)

# Training mode — DropConnect is applied.
y = layer(x, deterministic=False)

# Inference / validation — pass deterministic=True (default).
y = layer(x, deterministic=True)
```

The mask is resampled on every forward call. `weight_normalized=True`
on conv layers skips the per-step `‖W‖²` recomputation and pairs cleanly
with DropConnect.

---

## 10. Device & precision notes

- **Default device is the Metal GPU** on Apple Silicon. The first kernel
  launch incurs a small JIT cost; subsequent steps are fast.
- **fp32 is the default**. fp16 / bf16 reduce memory and speed up
  matmuls — bump `epsilon` to `1e-3` to stay numerically safe with
  small inputs.
- **CPU vs GPU output drift**: Metal matmul accumulates at lower
  precision than CPU fp32 (≈ 1e-3 element-wise on a small matmul).
  This is a property of the device, not the layer. The cross-framework
  parity tests pin to `mx.cpu` for that reason; user training runs are
  unaffected because optimization tolerates the noise.

---

## 11. Troubleshooting

| Symptom                                  | Likely cause                                      | Fix                                                                |
| ---------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------ |
| Gradients only flow through `alpha`      | Lazy params not built before `value_and_grad`     | Call the model once with a dummy input first (§ 8)                  |
| `NaN` loss in the first ~10 steps        | `epsilon` too small for the input/weight scale    | `YatNMN(features=..., epsilon=1e-3)`                                |
| `ValueError: padding must have length …` | Mixed scalar/tuple padding spec                   | Use a single scalar or a tuple matching the conv ndim               |
| Slow first step, fast after              | Metal kernel JIT                                  | Reuse the same Module across steps                                  |
| 3-D conv slower than expected            | First call is uncached                            | Warm up with a dummy `(1, *spatial, C)` input before benchmarking   |
| Tests assert tighter than ~1e-3 fail     | Comparing GPU output to CPU numpy reference       | Pin the run to `mx.cpu` for parity tests (see § 9)                  |

---

## 12. Next steps

- [Architecture & theory](../architecture.md)
- [Migration cheat sheet](../migration.md) — switching from PyTorch / NNX / TF
- [Cross-framework MNIST benchmark](../../website/blog-pages/20-cross-framework-mnist.html)
  — measured numbers for every backend
- [EXAMPLES.md](../../EXAMPLES.md) — more MLX snippets
