"""Rotary YAT Attention for MLX.

Combines Rotary Position Embeddings (RoPE) with the YAT attention
formula:

    1. Apply RoPE: q' = RoPE(q, pos), k' = RoPE(k, pos)
    2. Compute YAT: softmax((q'·k')² / (‖q' − k'‖² + ε)) · V

Mirrors ``nmn.nnx.layers.attention.rotary_yat`` (without the autoregressive
KV-cache, Performer, and LayerNorm-QK paths — those are deferred to a
future PR).

References:
    - RoFormer (https://arxiv.org/abs/2104.09864)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .attention import yat_attention_weights


__all__ = [
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "rotary_yat_attention_weights",
    "rotary_yat_attention",
    "RotaryYatAttention",
]


DEFAULT_CONSTANT_ALPHA = math.sqrt(2.0)


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    dtype: mx.Dtype = mx.float32,
) -> Tuple[mx.array, mx.array]:
    """Precompute cosine and sine frequencies for RoPE.

    Args:
        dim: Per-head dimension (must be even).
        max_seq_len: Maximum sequence length to precompute.
        theta: Base for the frequency computation.
        dtype: Output dtype.

    Returns:
        (cos_freqs, sin_freqs), each of shape ``(max_seq_len, dim // 2)``.
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2)[: dim // 2] / dim))
    t = np.arange(max_seq_len)
    freqs = np.outer(t, inv_freq)
    return (
        mx.array(np.cos(freqs).astype(np.float32)).astype(dtype),
        mx.array(np.sin(freqs).astype(np.float32)).astype(dtype),
    )


def apply_rotary_emb(
    x: mx.array,
    freqs_cos: mx.array,
    freqs_sin: mx.array,
    position_offset: int = 0,
) -> mx.array:
    """Apply RoPE to a tensor of shape ``(..., seq_len, num_heads, head_dim)``.

    The rotation pair is taken from the even / odd interleaved channels
    of the last axis, matching the reference implementation.
    """
    seq_len = x.shape[-3]
    max_seq_len = freqs_cos.shape[0]
    if position_offset + seq_len > max_seq_len:
        raise ValueError(
            f"Rotary frequencies precomputed for max_seq_len={max_seq_len}, "
            f"but position_offset + seq_len = {position_offset + seq_len} "
            f"exceeds it."
        )

    cos = freqs_cos[position_offset:position_offset + seq_len]
    sin = freqs_sin[position_offset:position_offset + seq_len]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    # Reshape cos / sin to (1, seq_len, 1, dim//2) — broadcasts across batch
    # and heads.
    cos = mx.reshape(cos, (1, seq_len, 1, -1))
    sin = mx.reshape(sin, (1, seq_len, 1, -1))

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    # Interleave back: stack on a new last axis, then flatten the last
    # two dims.
    out = mx.stack([rotated_even, rotated_odd], axis=-1)
    return mx.reshape(out, x.shape)


def rotary_yat_attention_weights(
    query: mx.array,
    key: mx.array,
    freqs_cos: mx.array,
    freqs_sin: mx.array,
    mask: Optional[mx.array] = None,
    dropout_rate: float = 0.0,
    training: bool = False,
    epsilon: float = 1e-5,
    alpha: Optional[mx.array] = None,
    scale: Optional[float] = None,
    position_offset: int = 0,
) -> mx.array:
    """Softmax-normalized YAT attention weights after applying RoPE to Q/K."""
    q_rot = apply_rotary_emb(query, freqs_cos, freqs_sin, position_offset)
    k_rot = apply_rotary_emb(key, freqs_cos, freqs_sin, position_offset)
    return yat_attention_weights(
        q_rot, k_rot,
        mask=mask,
        dropout_rate=dropout_rate,
        training=training,
        epsilon=epsilon,
        alpha=alpha,
        scale=scale,
        spherical=False,
    )


def rotary_yat_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    freqs_cos: mx.array,
    freqs_sin: mx.array,
    mask: Optional[mx.array] = None,
    dropout_rate: float = 0.0,
    training: bool = False,
    epsilon: float = 1e-5,
    alpha: Optional[mx.array] = None,
    scale: Optional[float] = None,
    position_offset: int = 0,
) -> mx.array:
    """RoPE → YAT attention → V."""
    weights = rotary_yat_attention_weights(
        query, key,
        freqs_cos, freqs_sin,
        mask=mask,
        dropout_rate=dropout_rate,
        training=training,
        epsilon=epsilon,
        alpha=alpha,
        scale=scale,
        position_offset=position_offset,
    )
    return mx.einsum("bhqk,bkhd->bqhd", weights, value)


class RotaryYatAttention(nn.Module):
    """Multi-head YAT attention with Rotary Position Embeddings.

    Architecture::

        Input → Linear(Q) → RoPE ─┐
        Input → Linear(K) → RoPE ─┼─→ yat_attention(Q', K', V) → Linear(out)
        Input → Linear(V) ────────┘

    The RoPE frequency tables are computed once at construction and
    cached as module attributes (``freqs_cos`` / ``freqs_sin``).

    Args:
        embed_dim: Total embedding dimension. Must be divisible by ``num_heads``.
        num_heads: Number of attention heads.
        max_seq_len: Maximum sequence length supported (defines the RoPE
            table size).
        theta: RoPE base frequency.
        dropout: Attention-dropout rate (only used in training mode).
        use_bias: Whether the projections carry a bias.
        use_alpha / constant_alpha: Same as ``MultiHeadYatAttention``.
        use_out_proj: Whether to apply the output projection.
        epsilon: YAT stability constant.
        dtype: MLX dtype for parameters and computation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int = 2048,
        *,
        theta: float = 10000.0,
        dropout: float = 0.0,
        use_bias: bool = True,
        use_alpha: bool = True,
        constant_alpha: Optional[Union[bool, float]] = None,
        use_out_proj: bool = True,
        epsilon: float = 1e-5,
        dtype: mx.Dtype = mx.float32,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        head_dim = embed_dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be even for RoPE."
            )
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.dropout = dropout
        self.use_bias = use_bias
        self.use_out_proj = use_out_proj
        self.epsilon = epsilon
        self.dtype = dtype

        # Alpha configuration.
        self._constant_alpha_value: Optional[float] = None
        if constant_alpha is not None and constant_alpha is not False:
            if constant_alpha is True:
                self._constant_alpha_value = DEFAULT_CONSTANT_ALPHA
            else:
                self._constant_alpha_value = float(constant_alpha)
            use_alpha = True
        elif use_alpha:
            self.alpha = mx.ones((1,), dtype=dtype)
        self.use_alpha = use_alpha
        self.constant_alpha = constant_alpha

        # RoPE frequency tables — cached on the module so MLX places them
        # on the active device.
        cos_freqs, sin_freqs = precompute_freqs_cis(head_dim, max_seq_len, theta, dtype)
        self.freqs_cos = cos_freqs
        self.freqs_sin = sin_freqs

        self.is_built = False

    def build(self, input_dim: int) -> None:
        if self.is_built:
            return
        std = math.sqrt(2.0 / (input_dim + self.embed_dim))

        def make_kernel(shape):
            return (mx.random.normal(shape=shape) * std).astype(self.dtype)

        self.q_kernel = make_kernel((input_dim, self.embed_dim))
        self.k_kernel = make_kernel((input_dim, self.embed_dim))
        self.v_kernel = make_kernel((input_dim, self.embed_dim))
        if self.use_bias:
            self.q_bias = mx.zeros((self.embed_dim,), dtype=self.dtype)
            self.k_bias = mx.zeros((self.embed_dim,), dtype=self.dtype)
            self.v_bias = mx.zeros((self.embed_dim,), dtype=self.dtype)

        if self.use_out_proj:
            self.out_kernel = make_kernel((self.embed_dim, self.embed_dim))
            if self.use_bias:
                self.out_bias = mx.zeros((self.embed_dim,), dtype=self.dtype)

        self.is_built = True

    def _linear(
        self, x: mx.array, kernel: mx.array, bias: Optional[mx.array]
    ) -> mx.array:
        y = x @ kernel
        if bias is not None:
            y = y + bias
        return y

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        *,
        training: bool = False,
        position_offset: int = 0,
    ) -> mx.array:
        if x.dtype != self.dtype:
            x = x.astype(self.dtype)
        if not self.is_built:
            self.build(int(x.shape[-1]))

        B, L, _ = x.shape
        if position_offset + L > self.max_seq_len:
            raise ValueError(
                f"position_offset + seq_len = {position_offset + L} exceeds "
                f"max_seq_len={self.max_seq_len}. Recreate the layer with a "
                f"larger max_seq_len."
            )

        q = self._linear(x, self.q_kernel, getattr(self, "q_bias", None))
        k = self._linear(x, self.k_kernel, getattr(self, "k_bias", None))
        v = self._linear(x, self.v_kernel, getattr(self, "v_bias", None))

        q = mx.reshape(q, (B, L, self.num_heads, self.head_dim))
        k = mx.reshape(k, (B, L, self.num_heads, self.head_dim))
        v = mx.reshape(v, (B, L, self.num_heads, self.head_dim))

        alpha_val = None
        scale_val = None
        if self.use_alpha:
            if self._constant_alpha_value is not None:
                scale_val = self._constant_alpha_value
            elif getattr(self, "alpha", None) is not None:
                alpha_val = self.alpha

        out = rotary_yat_attention(
            q, k, v,
            self.freqs_cos, self.freqs_sin,
            mask=mask,
            dropout_rate=self.dropout if training else 0.0,
            training=training,
            epsilon=self.epsilon,
            alpha=alpha_val,
            scale=scale_val,
            position_offset=position_offset,
        )

        out = mx.reshape(out, (B, L, self.embed_dim))
        if self.use_out_proj and getattr(self, "out_kernel", None) is not None:
            out = self._linear(out, self.out_kernel, getattr(self, "out_bias", None))
        return out
