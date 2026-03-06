from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array


@partial(jax.jit, static_argnames=("n",))
def soft_tanh(
    x: Array,
    n: float = 1.0,
) -> Array:
    """
    Maps a non-negative score to the range [-1, 1) using the soft-tanh function.

    The soft-tanh function is defined as:
    .. math::
        \\text{soft-tanh}_n(x) = \\frac{x^n - 1}{1 + x^n}

    The power `n` again controls the transition sharpness: higher `n` makes the
    function approach -1 more quickly for large `x`.

    Args:
        x (Array): A JAX array of non-negative scores (x >= 0).
        n (float, optional): The power to raise the score to. Defaults to 1.0.

    Returns:
        Array: The mapped scores in the range [-1, 1).

    Example:
        >>> import jax.numpy as jnp
        >>> from nmn.nnx.squashers import soft_tanh
        >>> x = jnp.array([0.0, 1.0, 2.0])
        >>> soft_tanh(x, n=1.0)
        Array([-1.        ,  0.        ,  0.33333334], dtype=float32)
    """
    if n <= 0:
        raise ValueError("Power 'n' must be positive.")

    x_n = jnp.power(x, n)
    return (x_n - 1.0) / (1.0 + x_n)
