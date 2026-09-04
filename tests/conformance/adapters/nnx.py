"""Flax NNX dense conformance adapter."""

from __future__ import annotations

import importlib.util

import numpy as np

from tests.conformance.oracle import DenseCase, OracleResult


class NnxAdapter:
    @staticmethod
    def _layer(case: DenseCase):
        import jax.numpy as jnp
        from flax import nnx

        from nmn.nnx import YatNMN

        layer = YatNMN(
            case.kernel.shape[0],
            case.kernel.shape[1],
            use_bias=True,
            use_alpha=True,
            epsilon=float(case.epsilon),
            learnable_epsilon=True,
            param_dtype=jnp.float32,
            rngs=nnx.Rngs(0),
        )
        layer.kernel[...] = jnp.asarray(case.kernel, dtype=jnp.float32)
        layer.bias[...] = jnp.asarray(case.bias, dtype=jnp.float32)
        layer.alpha[...] = jnp.asarray([float(case.alpha)], dtype=jnp.float32)
        return layer

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("flax") is not None

    @staticmethod
    def dense(case: DenseCase, *, compiled: bool = False) -> np.ndarray:
        import jax
        import jax.numpy as jnp

        layer = NnxAdapter._layer(case)
        function = jax.jit(layer) if compiled else layer
        return np.asarray(function(jnp.asarray(case.inputs, dtype=jnp.float32)))

    @staticmethod
    def dense_value_and_grad(
        case: DenseCase, *, compiled: bool = False
    ) -> OracleResult:
        import jax
        import jax.numpy as jnp

        layer = NnxAdapter._layer(case)
        inputs = jnp.asarray(case.inputs, dtype=jnp.float32)
        cotangent = jnp.asarray(case.cotangent, dtype=jnp.float32)

        def loss(model, values):
            output = model(values)
            return jnp.sum(output * cotangent), output

        function = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
        if compiled:
            function = jax.jit(function)
        (_, output), (parameter_grads, input_grad) = function(layer, inputs)
        raw_scale = jax.nn.sigmoid(layer.epsilon_param[...])
        gradients = {
            "input": input_grad,
            "kernel": parameter_grads.kernel[...],
            "bias": parameter_grads.bias[...],
            "alpha": jnp.squeeze(parameter_grads.alpha[...]),
            "epsilon": jnp.squeeze(parameter_grads.epsilon_param[...] / raw_scale),
        }
        return OracleResult(
            np.asarray(output),
            {name: np.asarray(value) for name, value in gradients.items()},
        )
