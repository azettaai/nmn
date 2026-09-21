# keras API reference

Generated from the public export lists and source signatures. Do not edit by hand.

Tensor annotations describe framework-native values; shapes, tracing, and dynamic
serialization settings remain runtime contracts. See [typing support](../typing.md).

## YatNMN

```python
YatNMN(units: int, use_bias: bool=True, constant_bias: float | bool | None=None, use_alpha: bool=True, constant_alpha: bool | float | None=None, positive_init: bool=False, epsilon: float=1e-05, learnable_epsilon: bool=False, spherical: bool=False, weight_normalized: bool=False, lazy: bool=False, freeze_kernel: bool | None=None, kernel_initializer: Any='glorot_normal', bias_initializer: Any='zeros', kernel_regularizer: Any=None, bias_regularizer: Any=None, activity_regularizer: Any=None, kernel_constraint: Any=None, bias_constraint: Any=None, **kwargs: Any)
```

A YAT densely-connected NN layer.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/nmn.py#L57)

## YatDense

```python
YatDense(units: int, use_bias: bool=True, constant_bias: float | bool | None=None, use_alpha: bool=True, constant_alpha: bool | float | None=None, positive_init: bool=False, epsilon: float=1e-05, learnable_epsilon: bool=False, spherical: bool=False, weight_normalized: bool=False, lazy: bool=False, freeze_kernel: bool | None=None, kernel_initializer: Any='glorot_normal', bias_initializer: Any='zeros', kernel_regularizer: Any=None, bias_regularizer: Any=None, activity_regularizer: Any=None, kernel_constraint: Any=None, bias_constraint: Any=None, **kwargs: Any)
```

A YAT densely-connected NN layer.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/nmn.py#L57)

## YatEmbed

```python
YatEmbed(num_embeddings: int, features: int, use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, epsilon: float=1e-05, spherical: bool=False, weight_normalized: bool=False, embedding_initializer: Any='glorot_normal', **kwargs: Any)
```

Embedding with YAT attend method (Keras Layer).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/embed.py#L31)

## MultiHeadYatAttention

```python
MultiHeadYatAttention(embed_dim: int, num_heads: int, dropout: float=0.0, use_bias: bool=True, use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, normalize_qk: bool=False, spherical: bool=False, use_out_proj: bool=True, epsilon: float=1e-05, kernel_initializer: str='glorot_normal', **kwargs: Any)
```

Multi-head attention using the YAT formula (Keras Layer).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/attention.py#L200)

## normalize_qk

```python
normalize_qk(query: Tensor, key: Tensor, epsilon: float=1e-06) -> tuple[Tensor, Tensor]
```

L2-normalizes query and key to unit vectors.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/attention.py#L34)

## yat_attention

```python
yat_attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None=None, dropout_rate: float=0.0, training: bool=False, epsilon: float=1e-05, alpha: Tensor | None=None, scale: Optional[float]=None, spherical: bool=False) -> Tensor
```

Computes YAT attention: softmax((Q.K)^2 / (||Q-K||^2 + eps)) . V

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/attention.py#L130)

## yat_attention_weights

```python
yat_attention_weights(query: Tensor, key: Tensor, mask: Tensor | None=None, dropout_rate: float=0.0, training: bool=False, epsilon: float=1e-05, alpha: Tensor | None=None, scale: Optional[float]=None, spherical: bool=False) -> Tensor
```

Computes YAT attention weights: softmax((Q.K)^2 / (||Q-K||^2 + eps))

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/attention.py#L43)

## yat_attention_normalized

```python
yat_attention_normalized(query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None=None, dropout_rate: float=0.0, training: bool=False, epsilon: float=1e-05, alpha: Tensor | None=None, scale: Optional[float]=None) -> Tensor
```

YAT attention with normalized Q/K (optimized).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/attention.py#L173)

## softermax

```python
softermax(x: Tensor, n: float=1.0, epsilon: float=1e-12, axis: Optional[int]=-1) -> Tensor
```

Normalizes non-negative scores using the Softermax function.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/squashers.py#L16)

## softer_sigmoid

```python
softer_sigmoid(x: Tensor, n: float=1.0) -> Tensor
```

Squashes non-negative scores into [0, 1) using soft-sigmoid.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/squashers.py#L39)

## soft_tanh

```python
soft_tanh(x: Tensor, n: float=1.0) -> Tensor
```

Maps non-negative scores to [-1, 1) using soft-tanh.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/squashers.py#L57)

## maclaurin_coeffs

```python
maclaurin_coeffs(b: float, eps: float, nmax: int) -> np.ndarray
```

Maclaurin coefficients ``a_n`` of ``kappa`` on the sphere.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/performer_yat.py#L97)

## create_maclaurin_projection

```python
create_maclaurin_projection(head_dim: int, num_features: int=256, bias: float=1.0, epsilon: float=1e-05, nmax: int=40, seed: Optional[int]=None, dtype: str='float32') -> Dict[str, Any]
```

Precompute the fixed projection for MAY (Random Maclaurin) features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/performer_yat.py#L119)

## maclaurin_features

```python
maclaurin_features(x: Tensor, params: Dict[str, Any], normalize: bool=True, epsilon: float=1e-06) -> Tensor
```

Compute MAY features ``phi(x) in R^M`` for every ``(..., d)`` token.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/performer_yat.py#L166)

## maclaurin_yat_attention

```python
maclaurin_yat_attention(query: Tensor, key: Tensor, value: Tensor, params: Dict[str, Any], causal: bool=False, epsilon: float=1e-09, normalize: bool=True) -> Tensor
```

Linear-complexity spherical YAT attention via MAY features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/performer_yat.py#L378)

## create_radial_projection

```python
create_radial_projection(head_dim: int, sketch_m: int=128, num_radial: int=8, radial_dim: int=64, bias: float=1.0, epsilon: float=1e-05, seed: Optional[int]=None, dtype: str='float32') -> Dict[str, Any]
```

Precompute the fixed projection for RAY (radial) features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/performer_yat.py#L218)

## radial_features

```python
radial_features(x: Tensor, params: Dict[str, Any], normalize: bool=True, epsilon: float=1e-06) -> Tensor
```

Compute RAY features ``phi(x)`` for every ``(..., d)`` token.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/performer_yat.py#L282)

## radial_yat_attention

```python
radial_yat_attention(query: Tensor, key: Tensor, value: Tensor, params: Dict[str, Any], causal: bool=False, epsilon: float=1e-09, normalize: bool=True) -> Tensor
```

Linear-complexity spherical YAT attention via RAY features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/performer_yat.py#L407)

## YatConv1D

```python
YatConv1D(filters: int, kernel_size: int | tuple[int, ...] | list[int], strides: int | tuple[int, ...] | list[int]=1, padding: str='valid', data_format: str | None=None, dilation_rate: int | tuple[int, ...] | list[int]=1, groups: int=1, use_bias: bool=True, constant_bias: float | bool | None=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, weight_normalized: bool=False, use_dropconnect: bool=False, drop_rate: float=0.0, tie_kernel_bank: bool=False, kernel_bank_size: int | None=None, kernel_bank_id: str='default', kernel_bank: Any=None, kernel_initializer: Any='orthogonal', bias_initializer: Any='zeros', kernel_regularizer: Any=None, bias_regularizer: Any=None, activity_regularizer: Any=None, kernel_constraint: Any=None, bias_constraint: Any=None, **kwargs: Any)
```

1D YAT convolution layer (e.g. temporal convolution).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/conv.py#L932)

## YatConv2D

```python
YatConv2D(filters: int, kernel_size: int | tuple[int, ...] | list[int], strides: int | tuple[int, ...] | list[int]=(1, 1), padding: str='valid', data_format: str | None=None, dilation_rate: int | tuple[int, ...] | list[int]=(1, 1), groups: int=1, use_bias: bool=True, constant_bias: float | bool | None=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, weight_normalized: bool=False, use_dropconnect: bool=False, drop_rate: float=0.0, tie_kernel_bank: bool=False, kernel_bank_size: int | None=None, kernel_bank_id: str='default', kernel_bank: Any=None, kernel_initializer: Any='orthogonal', bias_initializer: Any='zeros', kernel_regularizer: Any=None, bias_regularizer: Any=None, activity_regularizer: Any=None, kernel_constraint: Any=None, bias_constraint: Any=None, **kwargs: Any)
```

2D YAT convolution layer (e.g. spatial convolution over images).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/conv.py#L1062)

## YatConv3D

```python
YatConv3D(filters: int, kernel_size: int | tuple[int, ...] | list[int], strides: int | tuple[int, ...] | list[int]=(1, 1, 1), padding: str='valid', data_format: str | None=None, dilation_rate: int | tuple[int, ...] | list[int]=(1, 1, 1), groups: int=1, use_bias: bool=True, constant_bias: float | bool | None=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, weight_normalized: bool=False, use_dropconnect: bool=False, drop_rate: float=0.0, tie_kernel_bank: bool=False, kernel_bank_size: int | None=None, kernel_bank_id: str='default', kernel_bank: Any=None, kernel_initializer: Any='orthogonal', bias_initializer: Any='zeros', kernel_regularizer: Any=None, bias_regularizer: Any=None, activity_regularizer: Any=None, kernel_constraint: Any=None, bias_constraint: Any=None, **kwargs: Any)
```

3D YAT convolution layer (e.g. spatial convolution over volumes).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/conv.py#L1200)

## YatConv1d

```python
YatConv1d(filters: int, kernel_size: int | tuple[int, ...] | list[int], strides: int | tuple[int, ...] | list[int]=1, padding: str='valid', data_format: str | None=None, dilation_rate: int | tuple[int, ...] | list[int]=1, groups: int=1, use_bias: bool=True, constant_bias: float | bool | None=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, weight_normalized: bool=False, use_dropconnect: bool=False, drop_rate: float=0.0, tie_kernel_bank: bool=False, kernel_bank_size: int | None=None, kernel_bank_id: str='default', kernel_bank: Any=None, kernel_initializer: Any='orthogonal', bias_initializer: Any='zeros', kernel_regularizer: Any=None, bias_regularizer: Any=None, activity_regularizer: Any=None, kernel_constraint: Any=None, bias_constraint: Any=None, **kwargs: Any)
```

1D YAT convolution layer (e.g. temporal convolution).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/conv.py#L932)

## YatConv2d

```python
YatConv2d(filters: int, kernel_size: int | tuple[int, ...] | list[int], strides: int | tuple[int, ...] | list[int]=(1, 1), padding: str='valid', data_format: str | None=None, dilation_rate: int | tuple[int, ...] | list[int]=(1, 1), groups: int=1, use_bias: bool=True, constant_bias: float | bool | None=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, weight_normalized: bool=False, use_dropconnect: bool=False, drop_rate: float=0.0, tie_kernel_bank: bool=False, kernel_bank_size: int | None=None, kernel_bank_id: str='default', kernel_bank: Any=None, kernel_initializer: Any='orthogonal', bias_initializer: Any='zeros', kernel_regularizer: Any=None, bias_regularizer: Any=None, activity_regularizer: Any=None, kernel_constraint: Any=None, bias_constraint: Any=None, **kwargs: Any)
```

2D YAT convolution layer (e.g. spatial convolution over images).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/conv.py#L1062)

## YatConv3d

```python
YatConv3d(filters: int, kernel_size: int | tuple[int, ...] | list[int], strides: int | tuple[int, ...] | list[int]=(1, 1, 1), padding: str='valid', data_format: str | None=None, dilation_rate: int | tuple[int, ...] | list[int]=(1, 1, 1), groups: int=1, use_bias: bool=True, constant_bias: float | bool | None=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, weight_normalized: bool=False, use_dropconnect: bool=False, drop_rate: float=0.0, tie_kernel_bank: bool=False, kernel_bank_size: int | None=None, kernel_bank_id: str='default', kernel_bank: Any=None, kernel_initializer: Any='orthogonal', bias_initializer: Any='zeros', kernel_regularizer: Any=None, bias_regularizer: Any=None, activity_regularizer: Any=None, kernel_constraint: Any=None, bias_constraint: Any=None, **kwargs: Any)
```

3D YAT convolution layer (e.g. spatial convolution over volumes).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/conv.py#L1200)

## YatConvTranspose1D

```python
YatConvTranspose1D(filters: int, kernel_size: int | tuple[int, ...] | list[int], strides: int | tuple[int, ...] | list[int]=1, padding: str='valid', data_format: str | None=None, dilation_rate: int | tuple[int, ...] | list[int]=1, use_bias: bool=True, constant_bias: float | bool | None=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, weight_normalized: bool=False, use_dropconnect: bool=False, drop_rate: float=0.0, tie_kernel_bank: bool=False, kernel_bank_size: int | None=None, kernel_bank_id: str='default', kernel_bank: Any=None, kernel_initializer: Any='orthogonal', bias_initializer: Any='zeros', kernel_regularizer: Any=None, bias_regularizer: Any=None, activity_regularizer: Any=None, kernel_constraint: Any=None, bias_constraint: Any=None, output_padding: int | tuple[int, ...] | list[int] | None=None, output_shape_mode: str='framework', **kwargs: Any)
```

1D YAT transposed convolution layer (deconvolution).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/conv.py#L1305)

## YatConvTranspose2D

```python
YatConvTranspose2D(filters: int, kernel_size: int | tuple[int, ...] | list[int], strides: int | tuple[int, ...] | list[int]=(1, 1), padding: str='valid', data_format: str | None=None, dilation_rate: int | tuple[int, ...] | list[int]=(1, 1), use_bias: bool=True, constant_bias: float | bool | None=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, weight_normalized: bool=False, use_dropconnect: bool=False, drop_rate: float=0.0, tie_kernel_bank: bool=False, kernel_bank_size: int | None=None, kernel_bank_id: str='default', kernel_bank: Any=None, kernel_initializer: Any='orthogonal', bias_initializer: Any='zeros', kernel_regularizer: Any=None, bias_regularizer: Any=None, activity_regularizer: Any=None, kernel_constraint: Any=None, bias_constraint: Any=None, output_padding: int | tuple[int, ...] | list[int] | None=None, output_shape_mode: str='framework', **kwargs: Any)
```

2D YAT transposed convolution layer (deconvolution).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/conv.py#L1405)

## YatConvTranspose3D

```python
YatConvTranspose3D(filters: int, kernel_size: int | tuple[int, ...] | list[int], strides: int | tuple[int, ...] | list[int]=(1, 1, 1), padding: str='valid', data_format: str | None=None, dilation_rate: int | tuple[int, ...] | list[int]=(1, 1, 1), use_bias: bool=True, constant_bias: float | bool | None=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, weight_normalized: bool=False, use_dropconnect: bool=False, drop_rate: float=0.0, tie_kernel_bank: bool=False, kernel_bank_size: int | None=None, kernel_bank_id: str='default', kernel_bank: Any=None, kernel_initializer: Any='orthogonal', bias_initializer: Any='zeros', kernel_regularizer: Any=None, bias_regularizer: Any=None, activity_regularizer: Any=None, kernel_constraint: Any=None, bias_constraint: Any=None, output_padding: int | tuple[int, ...] | list[int] | None=None, output_shape_mode: str='framework', **kwargs: Any)
```

3D YAT transposed convolution layer (deconvolution).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/conv.py#L1505)

## YatConvTranspose1d

```python
YatConvTranspose1d(filters: int, kernel_size: int | tuple[int, ...] | list[int], strides: int | tuple[int, ...] | list[int]=1, padding: str='valid', data_format: str | None=None, dilation_rate: int | tuple[int, ...] | list[int]=1, use_bias: bool=True, constant_bias: float | bool | None=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, weight_normalized: bool=False, use_dropconnect: bool=False, drop_rate: float=0.0, tie_kernel_bank: bool=False, kernel_bank_size: int | None=None, kernel_bank_id: str='default', kernel_bank: Any=None, kernel_initializer: Any='orthogonal', bias_initializer: Any='zeros', kernel_regularizer: Any=None, bias_regularizer: Any=None, activity_regularizer: Any=None, kernel_constraint: Any=None, bias_constraint: Any=None, output_padding: int | tuple[int, ...] | list[int] | None=None, output_shape_mode: str='framework', **kwargs: Any)
```

1D YAT transposed convolution layer (deconvolution).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/conv.py#L1305)

## YatConvTranspose2d

```python
YatConvTranspose2d(filters: int, kernel_size: int | tuple[int, ...] | list[int], strides: int | tuple[int, ...] | list[int]=(1, 1), padding: str='valid', data_format: str | None=None, dilation_rate: int | tuple[int, ...] | list[int]=(1, 1), use_bias: bool=True, constant_bias: float | bool | None=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, weight_normalized: bool=False, use_dropconnect: bool=False, drop_rate: float=0.0, tie_kernel_bank: bool=False, kernel_bank_size: int | None=None, kernel_bank_id: str='default', kernel_bank: Any=None, kernel_initializer: Any='orthogonal', bias_initializer: Any='zeros', kernel_regularizer: Any=None, bias_regularizer: Any=None, activity_regularizer: Any=None, kernel_constraint: Any=None, bias_constraint: Any=None, output_padding: int | tuple[int, ...] | list[int] | None=None, output_shape_mode: str='framework', **kwargs: Any)
```

2D YAT transposed convolution layer (deconvolution).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/conv.py#L1405)

## YatConvTranspose3d

```python
YatConvTranspose3d(filters: int, kernel_size: int | tuple[int, ...] | list[int], strides: int | tuple[int, ...] | list[int]=(1, 1, 1), padding: str='valid', data_format: str | None=None, dilation_rate: int | tuple[int, ...] | list[int]=(1, 1, 1), use_bias: bool=True, constant_bias: float | bool | None=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, weight_normalized: bool=False, use_dropconnect: bool=False, drop_rate: float=0.0, tie_kernel_bank: bool=False, kernel_bank_size: int | None=None, kernel_bank_id: str='default', kernel_bank: Any=None, kernel_initializer: Any='orthogonal', bias_initializer: Any='zeros', kernel_regularizer: Any=None, bias_regularizer: Any=None, activity_regularizer: Any=None, kernel_constraint: Any=None, bias_constraint: Any=None, output_padding: int | tuple[int, ...] | list[int] | None=None, output_shape_mode: str='framework', **kwargs: Any)
```

3D YAT transposed convolution layer (deconvolution).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/keras/conv.py#L1505)
