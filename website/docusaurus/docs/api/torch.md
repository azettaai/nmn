# torch API reference

Generated from the public export lists and source signatures. Do not edit by hand.

Tensor annotations describe framework-native values; shapes, tracing, and dynamic
serialization settings remain runtime contracts. See [typing support](../typing.md).

## YatConv1D

```python
YatConv1D(in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t=1, padding: Union[str, _size_1_t]=0, dilation: _size_1_t=1, groups: int=1, bias: bool=True, constant_bias: Optional[float]=None, softplus_bias: bool=False, scalar_bias: bool=False, padding_mode: str='zeros', use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, use_dropconnect: bool=False, mask: Optional[Tensor]=None, epsilon: float=1e-05, learnable_epsilon: bool=False, drop_rate: float=0.0, weight_normalized: bool=False, tie_kernel_bank: bool=False, kernel_bank_size: Optional[int]=None, kernel_bank_id: str='default', device: torch.device | str | int | None=None, dtype: torch.dtype | None=None, param_dtype: torch.dtype | None=None, kernel_bank: Optional[KernelBank]=None)
```

1D YAT convolution layer implementing the YAT algorithm.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/layers/yat_conv1d.py#L30)

## YatConv2D

```python
YatConv2D(in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t=1, padding: Union[str, _size_2_t]=0, dilation: _size_2_t=1, groups: int=1, bias: bool=True, constant_bias: Optional[float]=None, softplus_bias: bool=False, scalar_bias: bool=False, padding_mode: str='zeros', use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, use_dropconnect: bool=False, mask: Optional[Tensor]=None, epsilon: float=1e-05, learnable_epsilon: bool=False, drop_rate: float=0.0, weight_normalized: bool=False, tie_kernel_bank: bool=False, kernel_bank_size: Optional[int]=None, kernel_bank_id: str='default', device: torch.device | str | int | None=None, dtype: torch.dtype | None=None, param_dtype: torch.dtype | None=None, kernel_bank: Optional[KernelBank]=None)
```

2D YAT convolution layer implementing the YAT algorithm.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/layers/yat_conv2d.py#L30)

## YatConv3D

```python
YatConv3D(in_channels: int, out_channels: int, kernel_size: _size_3_t, stride: _size_3_t=1, padding: Union[str, _size_3_t]=0, dilation: _size_3_t=1, groups: int=1, bias: bool=True, constant_bias: Optional[float]=None, softplus_bias: bool=False, scalar_bias: bool=False, padding_mode: str='zeros', use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, use_dropconnect: bool=False, mask: Optional[Tensor]=None, epsilon: float=1e-05, learnable_epsilon: bool=False, drop_rate: float=0.0, weight_normalized: bool=False, tie_kernel_bank: bool=False, kernel_bank_size: Optional[int]=None, kernel_bank_id: str='default', device: torch.device | str | int | None=None, dtype: torch.dtype | None=None, param_dtype: torch.dtype | None=None, kernel_bank: Optional[KernelBank]=None)
```

3D YAT convolution layer implementing the YAT algorithm.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/layers/yat_conv3d.py#L30)

## YatConvTranspose1D

```python
YatConvTranspose1D(in_channels: int, out_channels: int, kernel_size: _size_1_t, stride: _size_1_t=1, padding: _size_1_t=0, output_padding: _size_1_t=0, groups: int=1, bias: bool=True, constant_bias: Optional[float]=None, softplus_bias: bool=False, scalar_bias: bool=False, dilation: _size_1_t=1, padding_mode: str='zeros', use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, use_dropconnect: bool=False, mask: Optional[Tensor]=None, epsilon: float=1e-05, learnable_epsilon: bool=False, drop_rate: float=0.0, device: torch.device | str | int | None=None, dtype: torch.dtype | None=None, param_dtype: torch.dtype | None=None)
```

1D YAT transposed convolution layer implementing the YAT algorithm.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/layers/yat_conv_transpose1d.py#L28)

## YatConvTranspose2D

```python
YatConvTranspose2D(in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t=1, padding: _size_2_t=0, output_padding: _size_2_t=0, groups: int=1, bias: bool=True, constant_bias: Optional[float]=None, softplus_bias: bool=False, scalar_bias: bool=False, dilation: _size_2_t=1, padding_mode: str='zeros', use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, use_dropconnect: bool=False, mask: Optional[Tensor]=None, epsilon: float=1e-05, learnable_epsilon: bool=False, drop_rate: float=0.0, device: torch.device | str | int | None=None, dtype: torch.dtype | None=None, param_dtype: torch.dtype | None=None)
```

2D YAT transposed convolution layer implementing the YAT algorithm.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/layers/yat_conv_transpose2d.py#L28)

## YatConvTranspose3D

```python
YatConvTranspose3D(in_channels: int, out_channels: int, kernel_size: _size_3_t, stride: _size_3_t=1, padding: _size_3_t=0, output_padding: _size_3_t=0, groups: int=1, bias: bool=True, constant_bias: Optional[float]=None, softplus_bias: bool=False, scalar_bias: bool=False, dilation: _size_3_t=1, padding_mode: str='zeros', use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, use_dropconnect: bool=False, mask: Optional[Tensor]=None, epsilon: float=1e-05, learnable_epsilon: bool=False, drop_rate: float=0.0, device: torch.device | str | int | None=None, dtype: torch.dtype | None=None, param_dtype: torch.dtype | None=None)
```

3D YAT transposed convolution layer implementing the YAT algorithm.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/layers/yat_conv_transpose3d.py#L28)

## YatNMN

```python
YatNMN(in_features: int, out_features: int, bias: bool=True, constant_bias: Optional[float]=None, softplus_bias: bool=False, scalar_bias: bool=False, alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, dtype: Optional[torch.dtype]=None, param_dtype: torch.dtype=torch.float32, epsilon: float=1e-05, learnable_epsilon: bool=False, spherical: bool=False, positive_init: bool=False, weight_normalized: bool=False, lazy: bool=False, freeze_kernel: bool=False, tie_kernel_bank: bool=False, kernel_bank_size: Optional[int]=None, kernel_bank_id: str='default', kernel_init: Optional[Callable[[torch.Tensor], object]]=None, bias_init: Optional[Callable[[torch.Tensor], object]]=None, alpha_init: Optional[Callable[[torch.Tensor], object]]=None, device: torch.device | str | int | None=None, kernel_bank: Optional[KernelBank]=None)
```

  A PyTorch implementation of the Yat neuron with squared Euclidean distance transformation.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/nmn/yat_nmn.py#L28)

## KernelBank

```python
KernelBank()
```

Own parameters shared by explicitly associated YAT layers.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/kernel_bank.py#L13)

## MultiHeadYatAttention

```python
MultiHeadYatAttention(embed_dim: int, num_heads: int, dropout: float=0.0, bias: bool=True, use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, normalize_qk: bool=False, spherical: bool=False, use_out_proj: bool=True, epsilon: float=1e-05, device: torch.device | str | int | None=None, dtype: torch.dtype | None=None, param_dtype: torch.dtype | None=None)
```

Multi-head attention using the YAT formula.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/attention/multi_head.py#L35)

## yat_attention

```python
yat_attention(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor]=None, dropout_p: float=0.0, training: bool=True, epsilon: float=1e-05, alpha: Optional[Tensor]=None, scale: Optional[float]=None, spherical: bool=False) -> Tensor
```

Computes YAT attention: softmax((Q·K)² / (||Q-K||² + ε)) · V

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/attention/yat_attention.py#L167)

## yat_attention_weights

```python
yat_attention_weights(query: Tensor, key: Tensor, mask: Optional[Tensor]=None, dropout_p: float=0.0, training: bool=True, epsilon: float=1e-05, alpha: Optional[Tensor]=None, scale: Optional[float]=None, spherical: bool=False) -> Tensor
```

Computes YAT attention weights: softmax((Q·K)² / (||Q-K||² + ε))

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/attention/yat_attention.py#L60)

## create_maclaurin_projection

```python
create_maclaurin_projection(head_dim: int, num_features: int=256, bias: float=1.0, epsilon: float=1e-05, nmax: int=40, seed: Optional[int]=None, dtype: torch.dtype=torch.float32, device: Optional[torch.device]=None) -> Dict[str, Any]
```

Precompute the (fixed) MAY projection, shared by q and k.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/attention/performer_yat.py#L112)

## maclaurin_features

```python
maclaurin_features(x: Tensor, params: Dict[str, Any], normalize: bool=True, epsilon: float=1e-06) -> Tensor
```

Compute MAY features ``φ(x) ∈ ℝ^M`` for each token.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/attention/performer_yat.py#L171)

## maclaurin_yat_attention

```python
maclaurin_yat_attention(query: Tensor, key: Tensor, value: Tensor, params: Dict[str, Any], causal: bool=False, epsilon: float=1e-05) -> Tensor
```

Linear-complexity bias-aware spherical-YAT attention via MAY features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/attention/performer_yat.py#L213)

## create_radial_projection

```python
create_radial_projection(head_dim: int, sketch_m: int=128, num_radial: int=8, radial_dim: int=64, bias: float=1.0, epsilon: float=1e-05, seed: Optional[int]=None, dtype: torch.dtype=torch.float32, device: Optional[torch.device]=None) -> Dict[str, Any]
```

Precompute the (fixed) RAY projection, shared by q and k.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/attention/performer_yat.py#L247)

## radial_features

```python
radial_features(x: Tensor, params: Dict[str, Any], normalize: bool=True, epsilon: float=1e-06) -> Tensor
```

Compute RAY features ``φ(x) = psi(z) ⊗ φ_rad(z)`` for each token.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/attention/performer_yat.py#L319)

## radial_yat_attention

```python
radial_yat_attention(query: Tensor, key: Tensor, value: Tensor, params: Dict[str, Any], causal: bool=False, epsilon: float=1e-05) -> Tensor
```

Linear-complexity bias-aware spherical-YAT attention via RAY features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/attention/performer_yat.py#L372)

## YatEmbed

```python
YatEmbed(num_embeddings: int, features: int, use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, epsilon: float=1e-05, spherical: bool=False, weight_normalized: bool=False, device: torch.device | str | int | None=None, dtype: torch.dtype | None=None)
```

Embedding with YAT attend method.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/embed.py#L26)

## softermax

```python
softermax(x: Tensor, n: float=1.0, epsilon: float=1e-12, dim: Optional[int]=-1) -> Tensor
```

Normalizes non-negative scores using the Softermax function.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/squashers.py#L15)

## softer_sigmoid

```python
softer_sigmoid(x: Tensor, n: float=1.0) -> Tensor
```

Squashes non-negative scores into [0, 1) using soft-sigmoid.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/squashers.py#L41)

## soft_tanh

```python
soft_tanh(x: Tensor, n: float=1.0) -> Tensor
```

Maps non-negative scores to [-1, 1) using soft-tanh.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/torch/squashers.py#L62)
