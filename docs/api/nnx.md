# nnx API reference

Generated from the public export lists and source signatures. Do not edit by hand.

Tensor annotations describe framework-native values; shapes, tracing, and dynamic
serialization settings remain runtime contracts. See [typing support](../typing.md).

## YatNMN

```python
YatNMN(in_features: int, out_features: int, *, use_bias: bool=True, constant_bias: tp.Optional[float]=None, softplus_bias: bool=False, scalar_bias: bool=False, use_alpha: bool=True, constant_alpha: tp.Optional[tp.Union[bool, float]]=None, positive_init: bool=False, use_dropconnect: bool=False, fused: bool=False, dtype: tp.Optional[Dtype]=None, param_dtype: Dtype=jnp.float32, precision: PrecisionLike=None, compute_mode: str='fp32', distance_floor: float=0.0, kernel_init: Initializer=default_kernel_init, bias_init: Initializer=default_bias_init, alpha_init: Initializer=default_alpha_init, dot_general: DotGeneralT=lax.dot_general, promote_dtype: PromoteDtypeFn=dtypes.promote_dtype, epsilon: float=1e-05, learnable_epsilon: bool=False, spherical: bool=False, drop_rate: float=0.0, weight_normalized: bool=False, tie_kernel_bank: bool=False, kernel_bank_size: tp.Optional[int]=None, kernel_bank_id: str='default', kernel_bank: tp.Optional[KernelBank]=None, lazy: bool=False, freeze_kernel: tp.Optional[bool]=None, rngs: rnglib.Rngs)
```

A YAT linear transformation applied over the last dimension of the input.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/nmn.py#L56)

## FrozenParam

```python
FrozenParam()
```

A frozen (non-trainable) parameter variable.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/nmn.py#L40)

## KernelBank

```python
KernelBank()
```

Own parameters shared by explicitly associated NNX YAT layers.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/kernel_bank.py#L26)

## Embed

```python
Embed(num_embeddings: int, features: int, *, dtype: tp.Optional[Dtype]=None, param_dtype: Dtype=jnp.float32, embedding_init: Initializer=default_embed_init, promote_dtype: PromoteDtypeFn=dtypes.promote_dtype, use_alpha: bool=True, constant_alpha: tp.Optional[tp.Union[bool, float]]=None, epsilon: float=1e-05, learnable_epsilon: bool=False, spherical: bool=False, weight_normalized: bool=False, alpha_init: Initializer=default_alpha_init, rngs: rnglib.Rngs)
```

Embedding Module with YAT support.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/embed.py#L27)

## YatEmbed

```python
YatEmbed(num_embeddings: int, features: int, *, dtype: tp.Optional[Dtype]=None, param_dtype: Dtype=jnp.float32, embedding_init: Initializer=default_embed_init, promote_dtype: PromoteDtypeFn=dtypes.promote_dtype, use_alpha: bool=True, constant_alpha: tp.Optional[tp.Union[bool, float]]=None, epsilon: float=1e-05, learnable_epsilon: bool=False, spherical: bool=False, weight_normalized: bool=False, alpha_init: Initializer=default_alpha_init, rngs: rnglib.Rngs)
```

Embedding Module with YAT support.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/embed.py#L27)

## YatConv

```python
YatConv(in_features: int, out_features: int, kernel_size: int | tp.Sequence[int], strides: tp.Union[None, int, tp.Sequence[int]]=1, *, padding: PaddingLike='SAME', input_dilation: tp.Union[None, int, tp.Sequence[int]]=1, kernel_dilation: tp.Union[None, int, tp.Sequence[int]]=1, feature_group_count: int=1, use_bias: bool=True, constant_bias: tp.Optional[float]=None, softplus_bias: bool=False, scalar_bias: bool=False, use_alpha: bool=True, constant_alpha: tp.Optional[tp.Union[bool, float]]=None, use_dropconnect: bool=False, positive_init: bool=False, kernel_init: Initializer=default_kernel_init, bias_init: Initializer=default_bias_init, alpha_init: Initializer=default_alpha_init, mask: tp.Optional[Array]=None, dtype: tp.Optional[Dtype]=None, param_dtype: Dtype=jnp.float32, precision: PrecisionLike=None, conv_general_dilated: ConvGeneralDilatedT=lax.conv_general_dilated, promote_dtype: PromoteDtypeFn=dtypes.promote_dtype, epsilon: float=1e-05, learnable_epsilon: bool=False, drop_rate: float=0.0, weight_normalized: bool=False, tie_kernel_bank: bool=False, kernel_bank_size: tp.Optional[int]=None, kernel_bank_id: str='default', kernel_bank: tp.Optional[KernelBank]=None, rngs: rnglib.Rngs)
```

YAT Convolution Module wrapping ``lax.conv_general_dilated``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/conv/yat_conv.py#L52)

## YatConvTranspose

```python
YatConvTranspose(in_features: int, out_features: int, kernel_size: int | tp.Sequence[int], strides: int | tp.Sequence[int] | None=None, *, padding: PaddingLike='SAME', kernel_dilation: int | tp.Sequence[int] | None=None, output_padding: int | tp.Sequence[int] | None=None, use_bias: bool=True, constant_bias: tp.Optional[float]=None, softplus_bias: bool=False, scalar_bias: bool=False, use_alpha: bool=True, constant_alpha: tp.Optional[tp.Union[bool, float]]=None, use_dropconnect: bool=False, positive_init: bool=False, mask: Array | None=None, dtype: Dtype | None=None, param_dtype: Dtype=jnp.float32, precision: PrecisionLike | None=None, kernel_init: Initializer=default_kernel_init, bias_init: Initializer=default_bias_init, alpha_init: Initializer=default_alpha_init, transpose_kernel: bool=False, promote_dtype: PromoteDtypeFn=dtypes.promote_dtype, epsilon: float=1e-05, learnable_epsilon: bool=False, drop_rate: float=0.0, rngs: rnglib.Rngs)
```

YAT Transposed Convolution Module wrapping ``lax.conv_transpose``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/conv/yat_conv_transpose.py#L45)

## YatConv1D

```python
YatConv1D(in_features: int, out_features: int, kernel_size: int | tp.Sequence[int], strides: tp.Union[None, int, tp.Sequence[int]]=1, *, padding: PaddingLike='SAME', input_dilation: tp.Union[None, int, tp.Sequence[int]]=1, kernel_dilation: tp.Union[None, int, tp.Sequence[int]]=1, feature_group_count: int=1, use_bias: bool=True, constant_bias: tp.Optional[float]=None, softplus_bias: bool=False, scalar_bias: bool=False, use_alpha: bool=True, constant_alpha: tp.Optional[tp.Union[bool, float]]=None, use_dropconnect: bool=False, positive_init: bool=False, kernel_init: Initializer=default_kernel_init, bias_init: Initializer=default_bias_init, alpha_init: Initializer=default_alpha_init, mask: tp.Optional[Array]=None, dtype: tp.Optional[Dtype]=None, param_dtype: Dtype=jnp.float32, precision: PrecisionLike=None, conv_general_dilated: ConvGeneralDilatedT=lax.conv_general_dilated, promote_dtype: PromoteDtypeFn=dtypes.promote_dtype, epsilon: float=1e-05, learnable_epsilon: bool=False, drop_rate: float=0.0, weight_normalized: bool=False, tie_kernel_bank: bool=False, kernel_bank_size: tp.Optional[int]=None, kernel_bank_id: str='default', kernel_bank: tp.Optional[KernelBank]=None, rngs: rnglib.Rngs)
```

YAT Convolution Module wrapping ``lax.conv_general_dilated``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/conv/yat_conv.py#L52)

## YatConv2D

```python
YatConv2D(in_features: int, out_features: int, kernel_size: int | tp.Sequence[int], strides: tp.Union[None, int, tp.Sequence[int]]=1, *, padding: PaddingLike='SAME', input_dilation: tp.Union[None, int, tp.Sequence[int]]=1, kernel_dilation: tp.Union[None, int, tp.Sequence[int]]=1, feature_group_count: int=1, use_bias: bool=True, constant_bias: tp.Optional[float]=None, softplus_bias: bool=False, scalar_bias: bool=False, use_alpha: bool=True, constant_alpha: tp.Optional[tp.Union[bool, float]]=None, use_dropconnect: bool=False, positive_init: bool=False, kernel_init: Initializer=default_kernel_init, bias_init: Initializer=default_bias_init, alpha_init: Initializer=default_alpha_init, mask: tp.Optional[Array]=None, dtype: tp.Optional[Dtype]=None, param_dtype: Dtype=jnp.float32, precision: PrecisionLike=None, conv_general_dilated: ConvGeneralDilatedT=lax.conv_general_dilated, promote_dtype: PromoteDtypeFn=dtypes.promote_dtype, epsilon: float=1e-05, learnable_epsilon: bool=False, drop_rate: float=0.0, weight_normalized: bool=False, tie_kernel_bank: bool=False, kernel_bank_size: tp.Optional[int]=None, kernel_bank_id: str='default', kernel_bank: tp.Optional[KernelBank]=None, rngs: rnglib.Rngs)
```

YAT Convolution Module wrapping ``lax.conv_general_dilated``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/conv/yat_conv.py#L52)

## YatConv3D

```python
YatConv3D(in_features: int, out_features: int, kernel_size: int | tp.Sequence[int], strides: tp.Union[None, int, tp.Sequence[int]]=1, *, padding: PaddingLike='SAME', input_dilation: tp.Union[None, int, tp.Sequence[int]]=1, kernel_dilation: tp.Union[None, int, tp.Sequence[int]]=1, feature_group_count: int=1, use_bias: bool=True, constant_bias: tp.Optional[float]=None, softplus_bias: bool=False, scalar_bias: bool=False, use_alpha: bool=True, constant_alpha: tp.Optional[tp.Union[bool, float]]=None, use_dropconnect: bool=False, positive_init: bool=False, kernel_init: Initializer=default_kernel_init, bias_init: Initializer=default_bias_init, alpha_init: Initializer=default_alpha_init, mask: tp.Optional[Array]=None, dtype: tp.Optional[Dtype]=None, param_dtype: Dtype=jnp.float32, precision: PrecisionLike=None, conv_general_dilated: ConvGeneralDilatedT=lax.conv_general_dilated, promote_dtype: PromoteDtypeFn=dtypes.promote_dtype, epsilon: float=1e-05, learnable_epsilon: bool=False, drop_rate: float=0.0, weight_normalized: bool=False, tie_kernel_bank: bool=False, kernel_bank_size: tp.Optional[int]=None, kernel_bank_id: str='default', kernel_bank: tp.Optional[KernelBank]=None, rngs: rnglib.Rngs)
```

YAT Convolution Module wrapping ``lax.conv_general_dilated``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/conv/yat_conv.py#L52)

## YatConvTranspose1D

```python
YatConvTranspose1D(in_features: int, out_features: int, kernel_size: int | tp.Sequence[int], strides: int | tp.Sequence[int] | None=None, *, padding: PaddingLike='SAME', kernel_dilation: int | tp.Sequence[int] | None=None, output_padding: int | tp.Sequence[int] | None=None, use_bias: bool=True, constant_bias: tp.Optional[float]=None, softplus_bias: bool=False, scalar_bias: bool=False, use_alpha: bool=True, constant_alpha: tp.Optional[tp.Union[bool, float]]=None, use_dropconnect: bool=False, positive_init: bool=False, mask: Array | None=None, dtype: Dtype | None=None, param_dtype: Dtype=jnp.float32, precision: PrecisionLike | None=None, kernel_init: Initializer=default_kernel_init, bias_init: Initializer=default_bias_init, alpha_init: Initializer=default_alpha_init, transpose_kernel: bool=False, promote_dtype: PromoteDtypeFn=dtypes.promote_dtype, epsilon: float=1e-05, learnable_epsilon: bool=False, drop_rate: float=0.0, rngs: rnglib.Rngs)
```

YAT Transposed Convolution Module wrapping ``lax.conv_transpose``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/conv/yat_conv_transpose.py#L45)

## YatConvTranspose2D

```python
YatConvTranspose2D(in_features: int, out_features: int, kernel_size: int | tp.Sequence[int], strides: int | tp.Sequence[int] | None=None, *, padding: PaddingLike='SAME', kernel_dilation: int | tp.Sequence[int] | None=None, output_padding: int | tp.Sequence[int] | None=None, use_bias: bool=True, constant_bias: tp.Optional[float]=None, softplus_bias: bool=False, scalar_bias: bool=False, use_alpha: bool=True, constant_alpha: tp.Optional[tp.Union[bool, float]]=None, use_dropconnect: bool=False, positive_init: bool=False, mask: Array | None=None, dtype: Dtype | None=None, param_dtype: Dtype=jnp.float32, precision: PrecisionLike | None=None, kernel_init: Initializer=default_kernel_init, bias_init: Initializer=default_bias_init, alpha_init: Initializer=default_alpha_init, transpose_kernel: bool=False, promote_dtype: PromoteDtypeFn=dtypes.promote_dtype, epsilon: float=1e-05, learnable_epsilon: bool=False, drop_rate: float=0.0, rngs: rnglib.Rngs)
```

YAT Transposed Convolution Module wrapping ``lax.conv_transpose``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/conv/yat_conv_transpose.py#L45)

## YatConvTranspose3D

```python
YatConvTranspose3D(in_features: int, out_features: int, kernel_size: int | tp.Sequence[int], strides: int | tp.Sequence[int] | None=None, *, padding: PaddingLike='SAME', kernel_dilation: int | tp.Sequence[int] | None=None, output_padding: int | tp.Sequence[int] | None=None, use_bias: bool=True, constant_bias: tp.Optional[float]=None, softplus_bias: bool=False, scalar_bias: bool=False, use_alpha: bool=True, constant_alpha: tp.Optional[tp.Union[bool, float]]=None, use_dropconnect: bool=False, positive_init: bool=False, mask: Array | None=None, dtype: Dtype | None=None, param_dtype: Dtype=jnp.float32, precision: PrecisionLike | None=None, kernel_init: Initializer=default_kernel_init, bias_init: Initializer=default_bias_init, alpha_init: Initializer=default_alpha_init, transpose_kernel: bool=False, promote_dtype: PromoteDtypeFn=dtypes.promote_dtype, epsilon: float=1e-05, learnable_epsilon: bool=False, drop_rate: float=0.0, rngs: rnglib.Rngs)
```

YAT Transposed Convolution Module wrapping ``lax.conv_transpose``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/conv/yat_conv_transpose.py#L45)

## canonicalize_padding

```python
canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding
```

Canonicalizes conv padding to a jax.lax supported format.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/conv/utils.py#L25)

## conv_dimension_numbers

```python
conv_dimension_numbers(input_shape: tuple) -> lax.ConvDimensionNumbers
```

Computes the dimension numbers based on the input shape.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/conv/utils.py#L63)

## default_kernel_init

```python
default_kernel_init = initializers.xavier_normal()
```

default_kernel_init is defined in `src/nmn/nnx/layers/conv/utils.py`.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/conv/utils.py#L17)

## default_bias_init

```python
default_bias_init = initializers.zeros_init()
```

default_bias_init is defined in `src/nmn/nnx/layers/conv/utils.py`.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/conv/utils.py#L18)

## default_alpha_init

```python
default_alpha_init = initializers.ones_init()
```

default_alpha_init is defined in `src/nmn/nnx/layers/conv/utils.py`.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/conv/utils.py#L19)

## CONV_DEFAULT_CONSTANT_ALPHA

```python
CONV_DEFAULT_CONSTANT_ALPHA = jnp.sqrt(2.0)
```

CONV_DEFAULT_CONSTANT_ALPHA is defined in `src/nmn/nnx/layers/conv/utils.py`.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/conv/utils.py#L22)

## MultiHeadAttention

```python
MultiHeadAttention(num_heads: int, in_features: int, qkv_features: int | None=None, out_features: int | None=None, *, dtype: Dtype | None=None, param_dtype: Dtype=jnp.float32, broadcast_dropout: bool=True, dropout_rate: float=0.0, deterministic: bool | None=None, precision: PrecisionLike=None, kernel_init: Initializer=default_kernel_init, out_kernel_init: Initializer | None=None, bias_init: Initializer=initializers.zeros_init(), out_bias_init: Initializer | None=None, use_bias: bool=True, attention_fn: Callable[..., Array]=yat_attention, decode: bool | None=None, normalize_qk: bool=False, use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, alpha_init: Initializer=initializers.ones_init(), use_dropconnect: bool=False, dropconnect_rate: float=0.0, qkv_dot_general: DotGeneralT | None=None, out_dot_general: DotGeneralT | None=None, qkv_dot_general_cls: Any=None, out_dot_general_cls: Any=None, rngs: rnglib.Rngs, epsilon: float=1e-05, learnable_epsilon: bool=False, use_softermax: bool=False, power: float=1.0)
```

Multi-head attention with YAT or standard dot-product attention.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/multi_head.py#L132)

## MultiHeadYatAttention

```python
MultiHeadYatAttention(num_heads: int, in_features: int, qkv_features: int | None=None, out_features: int | None=None, *, dtype: Dtype | None=None, param_dtype: Dtype=jnp.float32, broadcast_dropout: bool=True, dropout_rate: float=0.0, deterministic: bool | None=None, precision: PrecisionLike=None, kernel_init: Initializer=default_kernel_init, out_kernel_init: Initializer | None=None, bias_init: Initializer=initializers.zeros_init(), out_bias_init: Initializer | None=None, use_bias: bool=True, attention_fn: Callable[..., Array]=yat_attention, decode: bool | None=None, normalize_qk: bool=False, use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, alpha_init: Initializer=initializers.ones_init(), use_dropconnect: bool=False, dropconnect_rate: float=0.0, qkv_dot_general: DotGeneralT | None=None, out_dot_general: DotGeneralT | None=None, qkv_dot_general_cls: Any=None, out_dot_general_cls: Any=None, rngs: rnglib.Rngs, epsilon: float=1e-05, learnable_epsilon: bool=False, use_softermax: bool=False, power: float=1.0)
```

Multi-head attention with YAT or standard dot-product attention.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/multi_head.py#L132)

## ATTENTION_DEFAULT_CONSTANT_ALPHA

```python
ATTENTION_DEFAULT_CONSTANT_ALPHA = jnp.sqrt(2.0)
```

ATTENTION_DEFAULT_CONSTANT_ALPHA is defined in `src/nmn/nnx/layers/attention/multi_head.py`.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/multi_head.py#L44)

## yat_attention

```python
yat_attention(query: Array, key: Array, value: Array, bias: Optional[Array]=None, mask: Optional[Array]=None, broadcast_dropout: bool=True, dropout_rng: Optional[Array]=None, dropout_rate: float=0.0, deterministic: bool=False, dtype: Optional[Dtype]=None, precision: PrecisionLike=None, module: Optional[Module]=None, epsilon: float=1e-05, use_softermax: bool=False, power: float=1.0, alpha: Optional[Array]=None, normalization: str='softmax') -> Array
```

Computes YAT attention: norm((Q·K)² / (||Q-K||² + ε)) · V

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/yat_attention.py#L176)

## yat_attention_weights

```python
yat_attention_weights(query: Array, key: Array, bias: Optional[Array]=None, mask: Optional[Array]=None, broadcast_dropout: bool=True, dropout_rng: Optional[Array]=None, dropout_rate: float=0.0, deterministic: bool=False, dtype: Optional[Dtype]=None, precision: PrecisionLike=None, module: Optional[Module]=None, epsilon: float=1e-05, use_softermax: bool=False, power: float=1.0, alpha: Optional[Array]=None, normalization: str='softmax') -> Array
```

Computes YAT attention weights: norm((Q·K)² / (||Q-K||² + ε))

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/yat_attention.py#L50)

## yat_attention_normalized

```python
yat_attention_normalized(query: Array, key: Array, value: Array, bias: Optional[Array]=None, mask: Optional[Array]=None, broadcast_dropout: bool=True, dropout_rng: Optional[Array]=None, dropout_rate: float=0.0, deterministic: bool=False, dtype: Optional[Dtype]=None, precision: PrecisionLike=None, module: Optional[Module]=None, epsilon: float=1e-05, use_softermax: bool=False, power: float=1.0, alpha: Optional[Array]=None) -> Array
```

Computes YAT attention with normalized Q and K (optimized).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/yat_attention.py#L280)

## yat_performer_attention

```python
yat_performer_attention(query: Array, key: Array, value: Array, projection: Array, bias: Optional[Array]=None, mask: Optional[Array]=None, broadcast_dropout: bool=True, dropout_rng: Optional[Array]=None, dropout_rate: float=0.0, deterministic: bool=False, dtype: Optional[Dtype]=None, precision: PrecisionLike=None, module: Optional[Module]=None, epsilon: float=1e-05, causal: bool=False, normalize_inputs: bool=True, alpha: Optional[Array]=None) -> Array
```

Computes YAT attention with Performer-style linear complexity.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/yat_attention.py#L441)

## yat_performer_feature_map

```python
yat_performer_feature_map(x: Array, projection: Array, epsilon: float=1e-06, pre_normalized: bool=False) -> Array
```

Applies YAT-adapted feature map for Performer approximation.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/yat_attention.py#L378)

## create_yat_projection

```python
create_yat_projection(key: Array, num_features: int, head_dim: int, dtype: Dtype=jnp.float32, orthogonal: bool=True) -> Array
```

Creates random projection matrix for YAT Performer.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/yat_attention.py#L583)

## normalize_qk

```python
normalize_qk(query: Array, key: Array, epsilon: float=1e-06) -> tuple[Array, Array]
```

Normalizes query and key to unit vectors.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/yat_attention.py#L253)

## RotaryYatAttention

```python
RotaryYatAttention(embed_dim: int, num_heads: int, max_seq_len: int=2048, *, theta: float=10000.0, dtype: Dtype | None=None, param_dtype: Dtype=jnp.float32, broadcast_dropout: bool=True, dropout_rate: float=0.0, precision: PrecisionLike=None, kernel_init: Initializer=default_kernel_init, bias_init: Initializer=initializers.zeros_init(), alpha_init: Initializer=initializers.ones_init(), use_bias: bool=False, normalize_qk: bool=False, use_out_proj: bool=True, epsilon: float=1e-05, use_softermax: bool=False, power: float=1.0, use_performer: bool=False, performer_kind: str='slay', num_anchor_features: int=16, num_prf_features: int=8, num_quad_nodes: int=1, performer_num_features: int=256, performer_bias: float=1.0, performer_nmax: int=40, performer_sketch_m: int=128, performer_num_radial: int=8, performer_radial_dim: int=64, causal: bool=False, performer_normalize: bool=True, use_alpha: bool=True, constant_alpha: Optional[Union[bool, float]]=None, learnable_epsilon: bool=False, normalization: str='softmax', rngs: rnglib.Rngs)
```

Multi-head Rotary YAT Attention with optional Performer mode.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/rotary_yat.py#L410)

## rotary_yat_attention

```python
rotary_yat_attention(query: Array, key: Array, value: Array, freqs_cos: Array, freqs_sin: Array, bias: Optional[Array]=None, mask: Optional[Array]=None, broadcast_dropout: bool=True, dropout_rng: Optional[Array]=None, dropout_rate: float=0.0, deterministic: bool=False, dtype: Optional[Dtype]=None, precision: PrecisionLike=None, module: Optional[Module]=None, epsilon: float=1e-05, use_softermax: bool=False, power: float=1.0, position_offset: int=0, alpha: Optional[Array]=None, normalization: str='softmax', key_position_offset: int | Array | None=None) -> Array
```

Computes Rotary YAT attention: RoPE + YAT formula + V weighted sum.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/rotary_yat.py#L264)

## rotary_yat_attention_weights

```python
rotary_yat_attention_weights(query: Array, key: Array, freqs_cos: Array, freqs_sin: Array, bias: Optional[Array]=None, mask: Optional[Array]=None, broadcast_dropout: bool=True, dropout_rng: Optional[Array]=None, dropout_rate: float=0.0, deterministic: bool=False, dtype: Optional[Dtype]=None, precision: PrecisionLike=None, module: Optional[Module]=None, epsilon: float=1e-05, use_softermax: bool=False, power: float=1.0, position_offset: int=0, alpha: Optional[Array]=None, normalization: str='softmax', key_position_offset: int | Array | None=None) -> Array
```

Computes Rotary YAT attention weights.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/rotary_yat.py#L192)

## rotary_yat_performer_attention

```python
rotary_yat_performer_attention(query: Array, key: Array, value: Array, freqs_cos: Array, freqs_sin: Array, performer_params: dict, bias: Optional[Array]=None, mask: Optional[Array]=None, broadcast_dropout: bool=True, dropout_rng: Optional[Array]=None, dropout_rate: float=0.0, deterministic: bool=False, dtype: Optional[Dtype]=None, precision: PrecisionLike=None, module: Optional[Module]=None, epsilon: float=1e-05, position_offset: int=0, causal: bool=False, normalize_inputs: bool=True, alpha: Optional[Array]=None, gradient_scaling: bool=True, key_position_offset: int | Array | None=None) -> Array
```

Computes Rotary YAT Performer attention using Multi-Scale TP-PRF.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/rotary_yat.py#L334)

## precompute_freqs_cis

```python
precompute_freqs_cis(dim: int, max_seq_len: int, theta: float=10000.0, dtype: Dtype=jnp.float32) -> Tuple[Array, Array]
```

Precomputes cosine and sine frequencies for RoPE.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/rotary_yat.py#L80)

## apply_rotary_emb

```python
apply_rotary_emb(x: Array, freqs_cos: Array, freqs_sin: Array, position_offset: int | Array=0) -> Array
```

Applies Rotary Position Embeddings to input tensor.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/rotary_yat.py#L117)

## yat_tp_attention

```python
yat_tp_attention(query: Array, key: Array, value: Array, params: dict, causal: bool=False, epsilon: float=1e-05, precision: PrecisionLike=None, gradient_scaling: bool=True, mask: Array | None=None) -> Array
```

Compute YAT attention using anchor-based tensor product features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/spherical_yat_performer.py#L281)

## yat_tp_features

```python
yat_tp_features(x: Array, params: dict, normalize: bool=True, epsilon: float=1e-06) -> Array
```

Compute anchor-based YAT features using tensor product.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/spherical_yat_performer.py#L202)

## create_yat_tp_projection

```python
create_yat_tp_projection(key: Array, head_dim: int, num_prf_features: int=8, num_quad_nodes: int=1, num_anchor_features: int=16, epsilon: float=1e-05, dtype: Dtype=jnp.float32) -> dict
```

Create parameters for anchor-based YAT attention.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/spherical_yat_performer.py#L83)

## create_maclaurin_projection

```python
create_maclaurin_projection(key: Array, head_dim: int, num_features: int=256, bias: float=1.0, epsilon: float=1e-05, nmax: int=40, dtype: Dtype=jnp.float32) -> dict
```

Create parameters for MAY (Random Maclaurin) Yat features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/maclaurin_yat.py#L107)

## maclaurin_features

```python
maclaurin_features(x: Array, params: dict, normalize: bool=True, epsilon: float=1e-06) -> Array
```

Compute MAY (Random Maclaurin) features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/maclaurin_yat.py#L169)

## maclaurin_yat_attention

```python
maclaurin_yat_attention(query: Array, key: Array, value: Array, params: dict, causal: bool=False, epsilon: float=1e-06, precision: PrecisionLike=None, gradient_scaling: bool=True, mask: Array | None=None) -> Array
```

Compute spherical Yat attention using MAY (Random Maclaurin) features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/maclaurin_yat.py#L298)

## maclaurin_coeffs

```python
maclaurin_coeffs(b: float, epsilon: float, nmax: int) -> np.ndarray
```

Maclaurin coefficients ``a_n`` of ``kappa(s) = (s+b)²/(C-2s)``.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/maclaurin_yat.py#L83)

## create_radial_projection

```python
create_radial_projection(key: Array, head_dim: int, sketch_m: int=128, num_radial: int=8, radial_dim: int=64, bias: float=1.0, epsilon: float=1e-05, dtype: Dtype=jnp.float32) -> dict
```

Create parameters for RAY (radial) Yat features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/radial_yat.py#L65)

## radial_features

```python
radial_features(x: Array, params: dict, normalize: bool=True, epsilon: float=1e-06) -> Array
```

Compute RAY (radial) features as the tensor product psi(z) ⊗ phi_rad(z).

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/radial_yat.py#L140)

## radial_yat_attention

```python
radial_yat_attention(query: Array, key: Array, value: Array, params: dict, causal: bool=False, epsilon: float=1e-06, precision: PrecisionLike=None, gradient_scaling: bool=True, mask: Array | None=None) -> Array
```

Compute spherical Yat attention using RAY (radial) features.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/radial_yat.py#L190)

## dot_product_attention

```python
dot_product_attention(query: Array, key: Array, value: Array, bias: Optional[Array]=None, mask: Optional[Array]=None, broadcast_dropout: bool=True, dropout_rng: Optional[Array]=None, dropout_rate: float=0.0, deterministic: bool=False, dtype: Optional[Dtype]=None, precision: PrecisionLike=None, module: Optional[Module]=None) -> Array
```

Computes scaled dot-product attention.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/dot_product.py#L93)

## dot_product_attention_weights

```python
dot_product_attention_weights(query: Array, key: Array, bias: Optional[Array]=None, mask: Optional[Array]=None, broadcast_dropout: bool=True, dropout_rng: Optional[Array]=None, dropout_rate: float=0.0, deterministic: bool=False, dtype: Optional[Dtype]=None, precision: PrecisionLike=None, module: Optional[Module]=None) -> Array
```

Computes scaled dot-product attention weights.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/dot_product.py#L29)

## make_attention_mask

```python
make_attention_mask(query_input: Array, key_input: Array, pairwise_fn: Callable[..., Any]=jnp.multiply, extra_batch_dims: int=0, dtype: Dtype=jnp.float32) -> Array
```

Creates an attention mask from query and key input arrays.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/masks.py#L21)

## make_causal_mask

```python
make_causal_mask(x: Array, extra_batch_dims: int=0, dtype: Dtype=jnp.float32) -> Array
```

Creates a causal (autoregressive) attention mask.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/masks.py#L63)

## combine_masks

```python
combine_masks(*masks: Optional[Array], dtype: Dtype=jnp.float32) -> Array | None
```

Combines multiple attention masks using logical AND.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/masks.py#L98)

## causal_attention_mask

```python
causal_attention_mask(seq_len: int) -> Array
```

Creates a simple lower triangular causal mask.

[Source](https://github.com/azettaai/nmn/blob/master/src/nmn/nnx/layers/attention/masks.py#L135)

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
