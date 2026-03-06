"""Neural-Matter Network (NMN) - Flax NNX Implementation.

This module provides YAT (You Are There) neural network layers and utilities
for Flax NNX, including:

- YatNMN: Core YAT linear transformation layer
- Embed: YAT-based embedding layer
- Attention: YAT attention mechanisms (standard, rotary, performer)
- Convolution: YAT convolution and transposed convolution layers
- RNN: YAT recurrent cell implementations (SimpleRNN, LSTM, GRU)
- Squashers: Custom activation functions (softermax, softer_sigmoid, soft_tanh)

Example:
    >>> from nmn.nnx import YatNMN, Embed, MultiHeadAttention, YatConv
    >>> from flax import nnx
    >>> import jax.numpy as jnp
    >>>
    >>> # Create a YAT linear layer
    >>> rngs = nnx.Rngs(0)
    >>> layer = YatNMN(in_features=128, out_features=256, constant_alpha=True, rngs=rngs)
    >>>
    >>> # Create a YAT embedding layer
    >>> embed = Embed(num_embeddings=1000, features=128, constant_alpha=True, rngs=rngs)
    >>>
    >>> # Create a multi-head YAT attention layer
    >>> attn = MultiHeadAttention(num_heads=8, in_features=128, rngs=rngs)
"""

# Core YAT layers
from nmn.nnx.nmn import YatNMN
from nmn.nnx.embed import Embed

# Attention mechanisms
from nmn.nnx.attention import (
    # YAT Attention
    yat_attention,
    yat_attention_weights,
    yat_attention_normalized,
    yat_performer_attention,
    yat_performer_feature_map,
    create_yat_projection,
    normalize_qk,
    # Rotary YAT Attention
    RotaryYatAttention,
    rotary_yat_attention,
    rotary_yat_attention_weights,
    rotary_yat_performer_attention,
    precompute_freqs_cis,
    apply_rotary_emb,
    # Spherical Yat-Performer
    yat_tp_attention,
    yat_tp_features,
    create_yat_tp_projection,
    # Standard Attention
    dot_product_attention,
    dot_product_attention_weights,
    # Multi-Head Attention
    MultiHeadAttention,
    DEFAULT_CONSTANT_ALPHA as ATTENTION_DEFAULT_CONSTANT_ALPHA,
    # Masks
    make_attention_mask,
    make_causal_mask,
    combine_masks,
    causal_attention_mask,
)

# Convolution layers
from nmn.nnx.conv import (
    YatConv,
    YatConvTranspose,
    canonicalize_padding,
    conv_dimension_numbers,
    default_kernel_init,
    default_bias_init,
    default_alpha_init,
    DEFAULT_CONSTANT_ALPHA as CONV_DEFAULT_CONSTANT_ALPHA,
)

# RNN layers
from nmn.nnx.rnn import (
    YatSimpleCell,
    YatLSTMCell,
    YatGRUCell,
    RNN,
    Bidirectional,
    RNNCellBase,
)

# Activation functions
from nmn.nnx.squashers import (
    softermax,
    softer_sigmoid,
    soft_tanh,
)

__all__ = [
    # Core layers
    "YatNMN",
    "Embed",
    
    # Attention - YAT
    "yat_attention",
    "yat_attention_weights",
    "yat_attention_normalized",
    "yat_performer_attention",
    "yat_performer_feature_map",
    "create_yat_projection",
    "normalize_qk",
    
    # Attention - Rotary YAT
    "RotaryYatAttention",
    "rotary_yat_attention",
    "rotary_yat_attention_weights",
    "rotary_yat_performer_attention",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    
    # Attention - Spherical Yat-Performer
    "yat_tp_attention",
    "yat_tp_features",
    "create_yat_tp_projection",
    
    # Attention - Standard
    "dot_product_attention",
    "dot_product_attention_weights",
    
    # Attention - Multi-Head
    "MultiHeadAttention",
    "ATTENTION_DEFAULT_CONSTANT_ALPHA",
    
    # Attention - Masks
    "make_attention_mask",
    "make_causal_mask",
    "combine_masks",
    "causal_attention_mask",
    
    # Convolution
    "YatConv",
    "YatConvTranspose",
    "canonicalize_padding",
    "conv_dimension_numbers",
    "default_kernel_init",
    "default_bias_init",
    "default_alpha_init",
    "CONV_DEFAULT_CONSTANT_ALPHA",
    
    # RNN
    "YatSimpleCell",
    "YatLSTMCell",
    "YatGRUCell",
    "RNN",
    "Bidirectional",
    "RNNCellBase",
    
    # Activations
    "softermax",
    "softer_sigmoid",
    "soft_tanh",
]
