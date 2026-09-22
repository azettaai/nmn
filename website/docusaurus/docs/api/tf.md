# tf API reference

Generated from the public export lists and source signatures. Do not edit by hand.

Tensor annotations describe framework-native values; shapes, tracing, and dynamic
serialization settings remain runtime contracts. See [typing support](../typing.md).

## YatNMN

```python
YatNMN(features: int, use_bias: bool=True, constant_bias: Optional[float]=None, use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, positive_init: bool=False, dtype: tf.DType=tf.float32, epsilon: float=1e-05, learnable_epsilon: bool=False, spherical: bool=False, weight_normalized: bool=False, return_weights: bool=False, lazy: bool=False, freeze_kernel: bool=False, name: Optional[str]=None)
```

Dense layer implementing the ⵟ-product (YAT) transformation.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/nmn.py#L23)

## YatDense

```python
YatDense(features: int, use_bias: bool=True, constant_bias: Optional[float]=None, use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, positive_init: bool=False, dtype: tf.DType=tf.float32, epsilon: float=1e-05, learnable_epsilon: bool=False, spherical: bool=False, weight_normalized: bool=False, return_weights: bool=False, lazy: bool=False, freeze_kernel: bool=False, name: Optional[str]=None)
```

Dense layer implementing the ⵟ-product (YAT) transformation.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/nmn.py#L23)

## YatEmbed

```python
YatEmbed(num_embeddings: int, features: int, use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, epsilon: float=1e-05, spherical: bool=False, weight_normalized: bool=False, dtype: tf.DType=tf.float32, name: Optional[str]=None)
```

Embedding with YAT attend method (tf.Module).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/embed.py#L24)

## MultiHeadYatAttention

```python
MultiHeadYatAttention(embed_dim: int, num_heads: int, dropout: float=0.0, use_bias: bool=True, use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, normalize_qk: bool=False, spherical: bool=False, use_out_proj: bool=True, epsilon: float=1e-05, dtype: tf.DType=tf.float32, name: Optional[str]=None)
```

Multi-head attention using the YAT formula (tf.Module).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/attention.py#L224)

## normalize_qk

```python
normalize_qk(query: tf.Tensor, key: tf.Tensor, epsilon: float=1e-06) -> Tuple[tf.Tensor, tf.Tensor]
```

L2-normalizes query and key to unit vectors.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/attention.py#L32)

## yat_attention

```python
yat_attention(query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, mask: Optional[tf.Tensor]=None, dropout_rate: float=0.0, training: bool=False, epsilon: float=1e-05, alpha: Optional[tf.Tensor]=None, scale: Optional[float]=None, spherical: bool=False) -> tf.Tensor
```

Computes YAT attention: softmax((Q.K)^2 / (||Q-K||^2 + eps)) . V

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/attention.py#L137)

## yat_attention_weights

```python
yat_attention_weights(query: tf.Tensor, key: tf.Tensor, mask: Optional[tf.Tensor]=None, dropout_rate: float=0.0, training: bool=False, epsilon: float=1e-05, alpha: Optional[tf.Tensor]=None, scale: Optional[float]=None, spherical: bool=False) -> tf.Tensor
```

Computes YAT attention weights: softmax((Q.K)^2 / (||Q-K||^2 + eps))

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/attention.py#L46)

## yat_attention_normalized

```python
yat_attention_normalized(query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, mask: Optional[tf.Tensor]=None, dropout_rate: float=0.0, training: bool=False, epsilon: float=1e-05, alpha: Optional[tf.Tensor]=None, scale: Optional[float]=None) -> tf.Tensor
```

YAT attention with normalized Q/K (optimized).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/attention.py#L180)

## softermax

```python
softermax(x: tf.Tensor, n: float=1.0, epsilon: float=1e-12, axis: Optional[int]=-1) -> tf.Tensor
```

Normalizes non-negative scores using the Softermax function.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/squashers.py#L15)

## softer_sigmoid

```python
softer_sigmoid(x: tf.Tensor, n: float=1.0) -> tf.Tensor
```

Squashes non-negative scores into [0, 1) using soft-sigmoid.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/squashers.py#L40)

## soft_tanh

```python
soft_tanh(x: tf.Tensor, n: float=1.0) -> tf.Tensor
```

Maps non-negative scores to [-1, 1) using soft-tanh.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/squashers.py#L60)

## maclaurin_coeffs

```python
maclaurin_coeffs(b: float, eps: float, nmax: int) -> np.ndarray
```

Maclaurin coefficients of ``kappa`` on the sphere.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/performer_yat.py#L79)

## create_maclaurin_projection

```python
create_maclaurin_projection(head_dim: int, num_features: int=256, bias: float=1.0, epsilon: float=1e-05, nmax: int=40, dtype: tf.DType=tf.float32, seed: Optional[int]=None) -> Dict[str, Any]
```

Precompute the fixed projection for MAY (Random Maclaurin) features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/performer_yat.py#L106)

## maclaurin_features

```python
maclaurin_features(x: tf.Tensor, params: Dict[str, Any], normalize: bool=True, epsilon: float=1e-06) -> tf.Tensor
```

Compute MAY features ``phi(x) ∈ ℝ^M`` for each token.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/performer_yat.py#L153)

## maclaurin_yat_attention

```python
maclaurin_yat_attention(query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, params: Dict[str, Any], causal: bool=False, epsilon: float=1e-05) -> tf.Tensor
```

Linear-complexity spherical-YAT attention via MAY features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/performer_yat.py#L202)

## create_radial_projection

```python
create_radial_projection(head_dim: int, sketch_m: int=128, num_radial: int=8, radial_dim: int=64, bias: float=1.0, epsilon: float=1e-05, dtype: tf.DType=tf.float32, seed: Optional[int]=None) -> Dict[str, Any]
```

Precompute the fixed projection for RAY (radial) features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/performer_yat.py#L233)

## radial_features

```python
radial_features(x: tf.Tensor, params: Dict[str, Any], normalize: bool=True, epsilon: float=1e-06) -> tf.Tensor
```

Compute RAY features for each token: ``psi(z) ⊗ phi_rad(z)``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/performer_yat.py#L299)

## radial_yat_attention

```python
radial_yat_attention(query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, params: Dict[str, Any], causal: bool=False, epsilon: float=1e-05) -> tf.Tensor
```

Linear-complexity spherical-YAT attention via RAY (radial) features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/performer_yat.py#L358)

## YatConv1D

```python
YatConv1D(filters: int, kernel_size: int, strides: int=1, padding: str='valid', dilation_rate: int=1, groups: int=1, use_bias: bool=True, constant_bias: Optional[float]=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, dtype: tf.DType=tf.float32, name: Optional[str]=None)
```

1D YAT convolution module using TensorFlow operations.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/conv.py#L517)

## YatConv2D

```python
YatConv2D(filters: int, kernel_size: Union[int, Tuple[int, int]], strides: Union[int, Tuple[int, int]]=(1, 1), padding: str='valid', dilation_rate: Union[int, Tuple[int, int]]=(1, 1), groups: int=1, use_bias: bool=True, constant_bias: Optional[float]=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, dtype: tf.DType=tf.float32, name: Optional[str]=None)
```

2D YAT convolution module using TensorFlow operations.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/conv.py#L578)

## YatConv3D

```python
YatConv3D(filters: int, kernel_size: Union[int, Tuple[int, int, int]], strides: Union[int, Tuple[int, int, int]]=(1, 1, 1), padding: str='valid', dilation_rate: Union[int, Tuple[int, int, int]]=(1, 1, 1), groups: int=1, use_bias: bool=True, constant_bias: Optional[float]=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, dtype: tf.DType=tf.float32, name: Optional[str]=None)
```

3D YAT convolution module using TensorFlow operations.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/conv.py#L642)

## YatConv1d

```python
YatConv1d(filters: int, kernel_size: int, strides: int=1, padding: str='valid', dilation_rate: int=1, groups: int=1, use_bias: bool=True, constant_bias: Optional[float]=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, dtype: tf.DType=tf.float32, name: Optional[str]=None)
```

1D YAT convolution module using TensorFlow operations.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/conv.py#L517)

## YatConv2d

```python
YatConv2d(filters: int, kernel_size: Union[int, Tuple[int, int]], strides: Union[int, Tuple[int, int]]=(1, 1), padding: str='valid', dilation_rate: Union[int, Tuple[int, int]]=(1, 1), groups: int=1, use_bias: bool=True, constant_bias: Optional[float]=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, dtype: tf.DType=tf.float32, name: Optional[str]=None)
```

2D YAT convolution module using TensorFlow operations.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/conv.py#L578)

## YatConv3d

```python
YatConv3d(filters: int, kernel_size: Union[int, Tuple[int, int, int]], strides: Union[int, Tuple[int, int, int]]=(1, 1, 1), padding: str='valid', dilation_rate: Union[int, Tuple[int, int, int]]=(1, 1, 1), groups: int=1, use_bias: bool=True, constant_bias: Optional[float]=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, dtype: tf.DType=tf.float32, name: Optional[str]=None)
```

3D YAT convolution module using TensorFlow operations.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/conv.py#L642)

## YatConvTranspose1D

```python
YatConvTranspose1D(filters: int, kernel_size: int, strides: int=1, padding: str='same', use_bias: bool=True, constant_bias: Optional[float]=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, dtype: tf.DType=tf.float32, name: Optional[str]=None, *, dilation_rate: int=1, output_padding: Optional[int]=None)
```

1D YAT transposed convolution (deconvolution) module using TensorFlow operations.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/conv.py#L706)

## YatConvTranspose2D

```python
YatConvTranspose2D(filters: int, kernel_size: Union[int, Tuple[int, int]], strides: Union[int, Tuple[int, int]]=(1, 1), padding: str='same', use_bias: bool=True, constant_bias: Optional[float]=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, dtype: tf.DType=tf.float32, name: Optional[str]=None, *, dilation_rate: Union[int, Tuple[int, int]]=(1, 1), output_padding: Optional[Union[int, Tuple[int, int]]]=None)
```

2D YAT transposed convolution (deconvolution) module using TensorFlow operations.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/conv.py#L768)

## YatConvTranspose3D

```python
YatConvTranspose3D(filters: int, kernel_size: Union[int, Tuple[int, int, int]], strides: Union[int, Tuple[int, int, int]]=(1, 1, 1), padding: str='same', use_bias: bool=True, constant_bias: Optional[float]=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, dtype: tf.DType=tf.float32, name: Optional[str]=None, *, dilation_rate: Union[int, Tuple[int, int, int]]=(1, 1, 1), output_padding: Optional[Union[int, Tuple[int, int, int]]]=None)
```

3D YAT transposed convolution (deconvolution) module using TensorFlow operations.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/conv.py#L830)

## YatConvTranspose1d

```python
YatConvTranspose1d(filters: int, kernel_size: int, strides: int=1, padding: str='same', use_bias: bool=True, constant_bias: Optional[float]=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, dtype: tf.DType=tf.float32, name: Optional[str]=None, *, dilation_rate: int=1, output_padding: Optional[int]=None)
```

1D YAT transposed convolution (deconvolution) module using TensorFlow operations.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/conv.py#L706)

## YatConvTranspose2d

```python
YatConvTranspose2d(filters: int, kernel_size: Union[int, Tuple[int, int]], strides: Union[int, Tuple[int, int]]=(1, 1), padding: str='same', use_bias: bool=True, constant_bias: Optional[float]=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, dtype: tf.DType=tf.float32, name: Optional[str]=None, *, dilation_rate: Union[int, Tuple[int, int]]=(1, 1), output_padding: Optional[Union[int, Tuple[int, int]]]=None)
```

2D YAT transposed convolution (deconvolution) module using TensorFlow operations.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/conv.py#L768)

## YatConvTranspose3d

```python
YatConvTranspose3d(filters: int, kernel_size: Union[int, Tuple[int, int, int]], strides: Union[int, Tuple[int, int, int]]=(1, 1, 1), padding: str='same', use_bias: bool=True, constant_bias: Optional[float]=None, use_alpha: bool=True, epsilon: float=1e-05, learnable_epsilon: bool=False, dtype: tf.DType=tf.float32, name: Optional[str]=None, *, dilation_rate: Union[int, Tuple[int, int, int]]=(1, 1, 1), output_padding: Optional[Union[int, Tuple[int, int, int]]]=None)
```

3D YAT transposed convolution (deconvolution) module using TensorFlow operations.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/tf/conv.py#L830)
