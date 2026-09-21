# linen API reference

Generated from the public export lists and source signatures. Do not edit by hand.

Tensor annotations describe framework-native values; shapes, tracing, and dynamic
serialization settings remain runtime contracts. See [typing support](../typing.md).

## YatNMN

```python
YatNMN(features: int, use_bias: bool, constant_bias: Optional[float], use_alpha: bool, constant_alpha: Optional[Any], positive_init: bool, dtype: Optional[Any], param_dtype: Any, precision: Any, kernel_init: Any, bias_init: Any, alpha_init: Any, epsilon: float, learnable_epsilon: bool, spherical: bool, weight_normalized: bool, lazy: bool, freeze_kernel: bool, dot_general: DotGeneralT | None, dot_general_cls: Any, return_weights: bool)
```

A custom transformation applied over the last dimension of the input using squared Euclidean distance.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/nmn.py#L41)

## YatEmbed

```python
YatEmbed(num_embeddings: int, features: int, use_alpha: bool, constant_alpha: Optional[Any], epsilon: float, spherical: bool, weight_normalized: bool, dtype: Optional[Any], param_dtype: Any, embedding_init: Any, alpha_init: Any)
```

Embedding with YAT attend method (Flax Linen).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/embed.py#L30)

## MultiHeadAttention

```python
MultiHeadAttention(num_heads: int, qkv_features: Optional[int], out_features: Optional[int], dropout_rate: float, use_bias: bool, use_alpha: bool, constant_alpha: Optional[Any], normalize_qk: bool, spherical: bool, epsilon: float, dtype: Optional[Any], param_dtype: Any, kernel_init: Any, bias_init: Any, alpha_init: Any, normalization: str)
```

Multi-head attention using the YAT formula (Flax Linen).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/attention.py#L169)

## normalize_qk

```python
normalize_qk(query: Array, key: Array, epsilon: float=1e-06) -> tuple[Array, Array]
```

Normalizes query and key to unit vectors.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/yat_attention.py#L253)

## yat_attention

```python
yat_attention(query: Array, key: Array, value: Array, mask: Optional[Array]=None, dropout_rate: float=0.0, deterministic: bool=True, epsilon: float=1e-05, alpha: Optional[Array]=None, spherical: bool=False, bias: Optional[Array]=None, normalization: str='softmax') -> Array
```

Computes normalized YAT attention over values.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/attention.py#L93)

## yat_attention_weights

```python
yat_attention_weights(query: Array, key: Array, mask: Optional[Array]=None, dropout_rate: float=0.0, deterministic: bool=True, epsilon: float=1e-05, alpha: Optional[Array]=None, spherical: bool=False, bias: Optional[Array]=None, normalization: str='softmax') -> Array
```

Computes YAT attention weights.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/attention.py#L45)

## yat_attention_normalized

```python
yat_attention_normalized(query: Array, key: Array, value: Array, mask: Optional[Array]=None, dropout_rate: float=0.0, deterministic: bool=True, epsilon: float=1e-05, alpha: Optional[Array]=None, bias: Optional[Array]=None, normalization: str='softmax') -> Array
```

YAT attention with normalized Q/K (optimized).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/attention.py#L141)

## softermax

```python
softermax(x: Array, n: float=1.0, epsilon: float=1e-12, axis: Optional[int]=-1) -> Array
```

Normalizes a set of non-negative scores using the Softermax function.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/squashers/softermax.py#L10)

## softer_sigmoid

```python
softer_sigmoid(x: Array, n: float=1.0) -> Array
```

Squashes a non-negative score into the range [0, 1) using the soft-sigmoid function.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/squashers/softer_sigmoid.py#L10)

## soft_tanh

```python
soft_tanh(x: Array, n: float=1.0) -> Array
```

Maps a non-negative score to the range [-1, 1) using the soft-tanh function.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/squashers/soft_tanh.py#L10)

## spherical_kappa

```python
spherical_kappa(s: Array, b: float, eps: float=1e-05) -> Array
```

Exact spherical-YAT kernel ``kappa(s) = (s + b)^2 / (C - 2 s)``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/performer_yat.py#L105)

## maclaurin_coeffs

```python
maclaurin_coeffs(b: float, eps: float, nmax: int) -> Array
```

Maclaurin coefficients of ``kappa`` on the sphere.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/performer_yat.py#L128)

## create_maclaurin_projection

```python
create_maclaurin_projection(key: Array, head_dim: int, num_features: int=256, bias: float=1.0, epsilon: float=1e-05, nmax: int=40, dtype: Any=jnp.float32) -> dict[str, Any]
```

Build a (fixed) Random-Maclaurin projection shared by q and k.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/performer_yat.py#L157)

## maclaurin_features

```python
maclaurin_features(x: Array, params: dict[str, Any], normalize: bool=True, eps: float=1e-06) -> Array
```

Random-Maclaurin feature map ``phi(x) ∈ R^M``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/performer_yat.py#L206)

## maclaurin_yat_attention

```python
maclaurin_yat_attention(query: Array, key: Array, value: Array, params: dict[str, Any], causal: bool=False, eps_div: float=1e-06) -> Array
```

Linear-complexity bias-aware YAT attention via MAY features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/performer_yat.py#L414)

## create_radial_projection

```python
create_radial_projection(key: Array, head_dim: int, sketch_m: int=128, num_radial: int=8, radial_dim: int=64, bias: float=1.0, epsilon: float=1e-05, dtype: Any=jnp.float32) -> dict[str, Any]
```

Build a RAY projection (degree-2 sketch x radial RFF).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/performer_yat.py#L254)

## radial_features

```python
radial_features(x: Array, params: dict[str, Any], normalize: bool=True, eps: float=1e-06) -> Array
```

RAY feature map ``phi(x) = psi(z) ⊗ phi_rad(z)``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/performer_yat.py#L320)

## radial_yat_attention

```python
radial_yat_attention(query: Array, key: Array, value: Array, params: dict[str, Any], causal: bool=False, eps_div: float=1e-06) -> Array
```

Linear-complexity bias-aware YAT attention via RAY features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/performer_yat.py#L441)

## linear_attention

```python
linear_attention(phi_q: Array, phi_k: Array, value: Array, causal: bool=False, eps_div: float=1e-06) -> Array
```

Generic linear-attention readout from feature maps.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/performer_yat.py#L371)

## YatConv1D

```python
YatConv1D(features: int, kernel_size: Sequence[int], strides: Sequence[int], padding: Union[str, Sequence[Tuple[int, int]]], input_dilation: Sequence[int], kernel_dilation: Sequence[int], feature_group_count: int, use_bias: bool, constant_bias: Optional[float], use_alpha: bool, dtype: Optional[Any], param_dtype: Any, kernel_init: Any, bias_init: Any, alpha_init: Any, epsilon: float, learnable_epsilon: bool)
```

1D YAT convolution layer for Flax Linen.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/conv.py#L300)

## YatConv2D

```python
YatConv2D(features: int, kernel_size: Sequence[int], strides: Sequence[int], padding: Union[str, Sequence[Tuple[int, int]]], input_dilation: Sequence[int], kernel_dilation: Sequence[int], feature_group_count: int, use_bias: bool, constant_bias: Optional[float], use_alpha: bool, dtype: Optional[Any], param_dtype: Any, kernel_init: Any, bias_init: Any, alpha_init: Any, epsilon: float, learnable_epsilon: bool)
```

2D YAT convolution layer for Flax Linen.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/conv.py#L354)

## YatConv3D

```python
YatConv3D(features: int, kernel_size: Sequence[int], strides: Sequence[int], padding: Union[str, Sequence[Tuple[int, int]]], input_dilation: Sequence[int], kernel_dilation: Sequence[int], feature_group_count: int, use_bias: bool, constant_bias: Optional[float], use_alpha: bool, dtype: Optional[Any], param_dtype: Any, kernel_init: Any, bias_init: Any, alpha_init: Any, epsilon: float, learnable_epsilon: bool)
```

3D YAT convolution layer for Flax Linen.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/conv.py#L407)

## YatConv1d

```python
YatConv1d(features: int, kernel_size: Sequence[int], strides: Sequence[int], padding: Union[str, Sequence[Tuple[int, int]]], input_dilation: Sequence[int], kernel_dilation: Sequence[int], feature_group_count: int, use_bias: bool, constant_bias: Optional[float], use_alpha: bool, dtype: Optional[Any], param_dtype: Any, kernel_init: Any, bias_init: Any, alpha_init: Any, epsilon: float, learnable_epsilon: bool)
```

1D YAT convolution layer for Flax Linen.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/conv.py#L300)

## YatConv2d

```python
YatConv2d(features: int, kernel_size: Sequence[int], strides: Sequence[int], padding: Union[str, Sequence[Tuple[int, int]]], input_dilation: Sequence[int], kernel_dilation: Sequence[int], feature_group_count: int, use_bias: bool, constant_bias: Optional[float], use_alpha: bool, dtype: Optional[Any], param_dtype: Any, kernel_init: Any, bias_init: Any, alpha_init: Any, epsilon: float, learnable_epsilon: bool)
```

2D YAT convolution layer for Flax Linen.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/conv.py#L354)

## YatConv3d

```python
YatConv3d(features: int, kernel_size: Sequence[int], strides: Sequence[int], padding: Union[str, Sequence[Tuple[int, int]]], input_dilation: Sequence[int], kernel_dilation: Sequence[int], feature_group_count: int, use_bias: bool, constant_bias: Optional[float], use_alpha: bool, dtype: Optional[Any], param_dtype: Any, kernel_init: Any, bias_init: Any, alpha_init: Any, epsilon: float, learnable_epsilon: bool)
```

3D YAT convolution layer for Flax Linen.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/conv.py#L407)

## YatConvTranspose1D

```python
YatConvTranspose1D(features: int, kernel_size: Sequence[int], strides: Sequence[int], padding: Union[str, Sequence[Tuple[int, int]]], use_bias: bool, constant_bias: Optional[float], use_alpha: bool, dtype: Optional[Any], param_dtype: Any, kernel_init: Any, bias_init: Any, alpha_init: Any, epsilon: float, learnable_epsilon: bool, kernel_dilation: Sequence[int], output_padding: Optional[Union[int, Sequence[int]]])
```

1D YAT transposed convolution layer for Flax Linen.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/conv.py#L460)

## YatConvTranspose2D

```python
YatConvTranspose2D(features: int, kernel_size: Sequence[int], strides: Sequence[int], padding: Union[str, Sequence[Tuple[int, int]]], use_bias: bool, constant_bias: Optional[float], use_alpha: bool, dtype: Optional[Any], param_dtype: Any, kernel_init: Any, bias_init: Any, alpha_init: Any, epsilon: float, learnable_epsilon: bool, kernel_dilation: Sequence[int], output_padding: Optional[Union[int, Sequence[int]]])
```

2D YAT transposed convolution layer for Flax Linen.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/conv.py#L512)

## YatConvTranspose3D

```python
YatConvTranspose3D(features: int, kernel_size: Sequence[int], strides: Sequence[int], padding: Union[str, Sequence[Tuple[int, int]]], use_bias: bool, constant_bias: Optional[float], use_alpha: bool, dtype: Optional[Any], param_dtype: Any, kernel_init: Any, bias_init: Any, alpha_init: Any, epsilon: float, learnable_epsilon: bool, kernel_dilation: Sequence[int], output_padding: Optional[Union[int, Sequence[int]]])
```

3D YAT transposed convolution layer for Flax Linen.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/conv.py#L564)

## YatConvTranspose1d

```python
YatConvTranspose1d(features: int, kernel_size: Sequence[int], strides: Sequence[int], padding: Union[str, Sequence[Tuple[int, int]]], use_bias: bool, constant_bias: Optional[float], use_alpha: bool, dtype: Optional[Any], param_dtype: Any, kernel_init: Any, bias_init: Any, alpha_init: Any, epsilon: float, learnable_epsilon: bool, kernel_dilation: Sequence[int], output_padding: Optional[Union[int, Sequence[int]]])
```

1D YAT transposed convolution layer for Flax Linen.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/conv.py#L460)

## YatConvTranspose2d

```python
YatConvTranspose2d(features: int, kernel_size: Sequence[int], strides: Sequence[int], padding: Union[str, Sequence[Tuple[int, int]]], use_bias: bool, constant_bias: Optional[float], use_alpha: bool, dtype: Optional[Any], param_dtype: Any, kernel_init: Any, bias_init: Any, alpha_init: Any, epsilon: float, learnable_epsilon: bool, kernel_dilation: Sequence[int], output_padding: Optional[Union[int, Sequence[int]]])
```

2D YAT transposed convolution layer for Flax Linen.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/conv.py#L512)

## YatConvTranspose3d

```python
YatConvTranspose3d(features: int, kernel_size: Sequence[int], strides: Sequence[int], padding: Union[str, Sequence[Tuple[int, int]]], use_bias: bool, constant_bias: Optional[float], use_alpha: bool, dtype: Optional[Any], param_dtype: Any, kernel_init: Any, bias_init: Any, alpha_init: Any, epsilon: float, learnable_epsilon: bool, kernel_dilation: Sequence[int], output_padding: Optional[Union[int, Sequence[int]]])
```

3D YAT transposed convolution layer for Flax Linen.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/linen/conv.py#L564)
