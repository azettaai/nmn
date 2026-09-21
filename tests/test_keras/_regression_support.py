"""Regression coverage for Keras issues #55, #56, #73-#76 and #78."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib

import keras
import numpy as np
import pytest

from nmn.keras import (
    MultiHeadYatAttention,
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
    YatEmbed,
    yat_attention,
    yat_attention_weights,
)
from nmn.keras._yat_core import stable_yat_ratio

BACKEND = keras.backend.backend()


def _attention_value_and_gradients(query, key, value, mask):
    if BACKEND == "jax":
        import jax
        import jax.numpy as jnp

        apply = jax.jit(lambda q, k, v: yat_attention(q, k, v, mask=mask))
        output = apply(query, key, value)
        grads = jax.grad(lambda q, k, v: jnp.sum(apply(q, k, v)), (0, 1, 2))(
            query, key, value
        )
        return output, grads
    if BACKEND == "torch":
        torch = pytest.importorskip("torch")
        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)
        apply = torch.compile(
            lambda q, k, v: yat_attention(q, k, v, mask=mask), backend="eager"
        )
        output = apply(query, key, value)
        return output, torch.autograd.grad(output.sum(), (query, key, value))
    tf = pytest.importorskip("tensorflow")
    apply = tf.function(lambda q, k, v: yat_attention(q, k, v, mask=mask))
    with tf.GradientTape() as tape:
        tape.watch((query, key, value))
        output = apply(query, key, value)
        loss = tf.reduce_sum(output)
    return output, tape.gradient(loss, (query, key, value))


def tensor(value, dtype="float32"):
    return keras.ops.convert_to_tensor(np.asarray(value), dtype=dtype)


def to_numpy(value, dtype=None):
    array = keras.ops.convert_to_numpy(value)
    return np.asarray(array, dtype=dtype) if dtype is not None else array


def input_gradient(layer, value):
    if BACKEND == "jax":
        import jax
        import jax.numpy as jnp

        return jax.grad(lambda x: jnp.sum(layer(x)))(value)
    if BACKEND == "tensorflow":
        tf = pytest.importorskip("tensorflow")
        with tf.GradientTape() as tape:
            tape.watch(value)
            loss = tf.reduce_sum(layer(value))
        return tape.gradient(loss, value)
    pytest.skip("gradient assertion is implemented for JAX and TensorFlow backends")


def ops_cast(value, dtype):
    return keras.ops.cast(value, dtype)
