# mlx API reference

Generated from the public export lists and source signatures. Do not edit by hand.

Tensor annotations describe framework-native values; shapes, tracing, and dynamic
serialization settings remain runtime contracts. See [typing support](../typing.md).

## YatNMN

```python
YatNMN(features: int, use_bias: bool=True, constant_bias: Optional[float]=None, use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, positive_init: bool=False, use_dropconnect: bool=False, drop_rate: float=0.0, fused: bool=False, dtype: mx.Dtype=mx.float32, epsilon: float=1e-05, learnable_epsilon: bool=False, spherical: bool=False, weight_normalized: bool=False, return_weights: bool=False, lazy: bool=False, freeze_kernel: Optional[bool]=None)
```

Dense layer implementing the ⵟ-Product (YAT) transformation.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/nmn.py#L32)

## YatDense

```python
YatDense(features: int, use_bias: bool=True, constant_bias: Optional[float]=None, use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, positive_init: bool=False, use_dropconnect: bool=False, drop_rate: float=0.0, fused: bool=False, dtype: mx.Dtype=mx.float32, epsilon: float=1e-05, learnable_epsilon: bool=False, spherical: bool=False, weight_normalized: bool=False, return_weights: bool=False, lazy: bool=False, freeze_kernel: Optional[bool]=None)
```

Dense layer implementing the ⵟ-Product (YAT) transformation.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/nmn.py#L32)

## YatEmbed

```python
YatEmbed(num_embeddings: int, features: int, use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, epsilon: float=1e-05, spherical: bool=False, weight_normalized: bool=False, dtype: mx.Dtype=mx.float32)
```

Embedding table with a YAT-flavoured ``attend`` method.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/embed.py#L28)

## MultiHeadYatAttention

```python
MultiHeadYatAttention(embed_dim: int, num_heads: int, dropout: float=0.0, use_bias: bool=True, use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, normalize_qk: bool=False, spherical: bool=False, use_out_proj: bool=True, epsilon: float=1e-05, dtype: mx.Dtype=mx.float32)
```

Multi-head YAT attention.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/attention.py#L166)

## normalize_qk

```python
normalize_qk(query: mx.array, key: mx.array, epsilon: float=1e-06) -> Tuple[mx.array, mx.array]
```

L2-normalize ``query`` and ``key`` along the last axis.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/attention.py#L39)

## yat_attention

```python
yat_attention(query: mx.array, key: mx.array, value: mx.array, mask: Optional[mx.array]=None, dropout_rate: float=0.0, training: bool=False, epsilon: float=1e-05, alpha: Optional[mx.array]=None, scale: Optional[float]=None, spherical: bool=False) -> mx.array
```

Softmax(YAT score) @ V. See ``yat_attention_weights`` for conventions.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/attention.py#L113)

## yat_attention_weights

```python
yat_attention_weights(query: mx.array, key: mx.array, mask: Optional[mx.array]=None, dropout_rate: float=0.0, training: bool=False, epsilon: float=1e-05, alpha: Optional[mx.array]=None, scale: Optional[float]=None, spherical: bool=False) -> mx.array
```

Softmax over the YAT score.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/attention.py#L50)

## yat_attention_normalized

```python
yat_attention_normalized(query: mx.array, key: mx.array, value: mx.array, mask: Optional[mx.array]=None, dropout_rate: float=0.0, training: bool=False, epsilon: float=1e-05, alpha: Optional[mx.array]=None, scale: Optional[float]=None) -> mx.array
```

``yat_attention`` with the QK normalization shortcut (``spherical=True``).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/attention.py#L140)

## RotaryYatAttention

```python
RotaryYatAttention(embed_dim: int, num_heads: int, max_seq_len: int=2048, *, theta: float=10000.0, dropout: float=0.0, use_bias: bool=True, use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, use_out_proj: bool=True, epsilon: float=1e-05, dtype: mx.Dtype=mx.float32)
```

Multi-head YAT attention with Rotary Position Embeddings.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/rotary.py#L185)

## precompute_freqs_cis

```python
precompute_freqs_cis(dim: int, max_seq_len: int, theta: float=10000.0, dtype: mx.Dtype=mx.float32) -> Tuple[mx.array, mx.array]
```

Precompute cosine and sine frequencies for RoPE.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/rotary.py#L57)

## apply_rotary_emb

```python
apply_rotary_emb(x: mx.array, freqs_cos: mx.array, freqs_sin: mx.array, position_offset: int=0) -> mx.array
```

Apply RoPE to a tensor of shape ``(..., seq_len, num_heads, head_dim)``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/rotary.py#L85)

## rotary_yat_attention

```python
rotary_yat_attention(query: mx.array, key: mx.array, value: mx.array, freqs_cos: mx.array, freqs_sin: mx.array, mask: Optional[mx.array]=None, dropout_rate: float=0.0, training: bool=False, epsilon: float=1e-05, alpha: Optional[mx.array]=None, scale: Optional[float]=None, position_offset: int=0) -> mx.array
```

RoPE → YAT attention → V.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/rotary.py#L154)

## rotary_yat_attention_weights

```python
rotary_yat_attention_weights(query: mx.array, key: mx.array, freqs_cos: mx.array, freqs_sin: mx.array, mask: Optional[mx.array]=None, dropout_rate: float=0.0, training: bool=False, epsilon: float=1e-05, alpha: Optional[mx.array]=None, scale: Optional[float]=None, position_offset: int=0) -> mx.array
```

Softmax-normalized YAT attention weights after applying RoPE to Q/K.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/rotary.py#L125)

## create_yat_tp_projection

```python
create_yat_tp_projection(head_dim: int, num_prf_features: int=8, num_quad_nodes: int=1, num_anchor_features: int=16, epsilon: float=1e-05, dtype: mx.Dtype=mx.float32, seed: Optional[int]=None) -> dict[str, Any]
```

Precompute the parameters for anchor-based YAT attention.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/performer.py#L34)

## yat_tp_features

```python
yat_tp_features(x: mx.array, params: dict[str, Any], normalize: bool=True, epsilon: float=1e-06) -> mx.array
```

Compute ``φ(x) ∈ ℝ^(R·P·M)`` for each ``(seq, head)`` token.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/performer.py#L93)

## yat_tp_attention

```python
yat_tp_attention(query: mx.array, key: mx.array, value: mx.array, params: dict[str, Any], causal: bool=False, epsilon: float=1e-05) -> mx.array
```

Linear-complexity YAT attention via tensor-product features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/performer.py#L154)

## maclaurin_coeffs

```python
maclaurin_coeffs(b: float, epsilon: float, nmax: int) -> np.ndarray
```

Maclaurin coefficients of ``kappa`` on the sphere.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/may.py#L62)

## create_maclaurin_projection

```python
create_maclaurin_projection(head_dim: int, num_features: int=256, bias: float=1.0, epsilon: float=1e-05, nmax: int=40, dtype: mx.Dtype=mx.float32, seed: Optional[int]=None) -> dict[str, Any]
```

Precompute the (fixed) Random Maclaurin projection shared by q and k.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/may.py#L86)

## maclaurin_features

```python
maclaurin_features(x: mx.array, params: dict[str, Any], normalize: bool=True, epsilon: float=1e-06) -> mx.array
```

Compute the Random Maclaurin feature ``phi(x) ∈ ℝ^M`` per token.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/may.py#L138)

## maclaurin_yat_attention

```python
maclaurin_yat_attention(query: mx.array, key: mx.array, value: mx.array, params: dict[str, Any], causal: bool=False, epsilon: float=1e-05) -> mx.array
```

Linear-complexity *bias-aware* YAT attention via MAY features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/may.py#L181)

## create_radial_projection

```python
create_radial_projection(head_dim: int, sketch_m: int=8, num_radial: int=4, radial_dim: int=8, bias: float=1.0, epsilon: float=1e-05, dtype: mx.Dtype=mx.float32, seed: Optional[int]=None) -> dict[str, Any]
```

Precompute the (fixed) RAY projection shared by q and k.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/ray.py#L55)

## radial_features

```python
radial_features(x: mx.array, params: dict[str, Any], normalize: bool=True, epsilon: float=1e-06) -> mx.array
```

Compute the RAY feature ``psi(z) ⊗ phi_rad(z)`` per token.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/ray.py#L120)

## radial_yat_attention

```python
radial_yat_attention(query: mx.array, key: mx.array, value: mx.array, params: dict[str, Any], causal: bool=False, epsilon: float=1e-05) -> mx.array
```

Linear-complexity *bias-aware* YAT attention via RAY features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/ray.py#L174)

## GoatYatAttention

```python
GoatYatAttention(embed_dim: int, num_heads: int, *, variant: str='v', use_out_proj: bool=True, use_bias: bool=False, epsilon: float=1.0, dtype: mx.Dtype=mx.float32)
```

Multi-head GOAT self-attention (value-only or projection-free).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/goat.py#L172)

## goat_yat_attention

```python
goat_yat_attention(q: mx.array, k: mx.array, v: mx.array, b: mx.array, eps: mx.array, mask: Optional[mx.array]=None, self_mask: bool=True) -> mx.array
```

Apply GOAT weights to values.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/goat.py#L152)

## goat_yat_attention_weights

```python
goat_yat_attention_weights(q: mx.array, k: mx.array, b: mx.array, eps: mx.array, mask: Optional[mx.array]=None, self_mask: bool=True, floor: float=1e-08) -> mx.array
```

:math:`\ell_1`-normalised YAT-kernel attention weights on raw head slices.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/goat.py#L76)

## fused_yat_score

```python
fused_yat_score(x: mx.array, w: mx.array, bias: Optional[mx.array]=None, alpha: Optional[mx.array]=None, epsilon: Union[float, mx.array]=1e-05) -> mx.array
```

Convenience wrapper that fills in defaults and flattens batch dims.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/fused.py#L174)

## is_gpu_available

```python
is_gpu_available() -> bool
```

``True`` iff the current default device is the Metal GPU.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/fused.py#L64)

## softermax

```python
softermax(x: mx.array, n: float=1.0, epsilon: float=1e-12, axis: Optional[int]=-1) -> mx.array
```

Normalizes non-negative scores using the Softermax function.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/squashers.py#L15)

## softer_sigmoid

```python
softer_sigmoid(x: mx.array, n: float=1.0) -> mx.array
```

Squashes non-negative scores into [0, 1) using soft-sigmoid.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/squashers.py#L41)

## soft_tanh

```python
soft_tanh(x: mx.array, n: float=1.0) -> mx.array
```

Maps non-negative scores to [-1, 1) using soft-tanh.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/squashers.py#L52)

## YatConv1D

```python
YatConv1D()
```

1D YAT convolution. Input: ``(N, L, C_in)``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/conv.py#L434)

## YatConv2D

```python
YatConv2D()
```

2D YAT convolution. Input: ``(N, H, W, C_in)``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/conv.py#L440)

## YatConv3D

```python
YatConv3D()
```

3D YAT convolution. Input: ``(N, D, H, W, C_in)``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/conv.py#L446)

## YatConv1d

```python
YatConv1d()
```

1D YAT convolution. Input: ``(N, L, C_in)``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/conv.py#L434)

## YatConv2d

```python
YatConv2d()
```

2D YAT convolution. Input: ``(N, H, W, C_in)``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/conv.py#L440)

## YatConv3d

```python
YatConv3d()
```

3D YAT convolution. Input: ``(N, D, H, W, C_in)``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/conv.py#L446)

## YatConvTranspose1D

```python
YatConvTranspose1D()
```

1D YAT transposed convolution. Input: ``(N, L, C_in)``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/conv.py#L677)

## YatConvTranspose2D

```python
YatConvTranspose2D()
```

2D YAT transposed convolution. Input: ``(N, H, W, C_in)``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/conv.py#L683)

## YatConvTranspose3D

```python
YatConvTranspose3D()
```

3D YAT transposed convolution. Input: ``(N, D, H, W, C_in)``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/conv.py#L689)

## YatConvTranspose1d

```python
YatConvTranspose1d()
```

1D YAT transposed convolution. Input: ``(N, L, C_in)``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/conv.py#L677)

## YatConvTranspose2d

```python
YatConvTranspose2d()
```

2D YAT transposed convolution. Input: ``(N, H, W, C_in)``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/conv.py#L683)

## YatConvTranspose3d

```python
YatConvTranspose3d()
```

3D YAT transposed convolution. Input: ``(N, D, H, W, C_in)``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/mlx/conv.py#L689)
