"""Flax Linen dense conformance adapter."""

from __future__ import annotations

import importlib.util

import numpy as np

from tests.conformance.oracle import DenseCase, OracleResult


class LinenAdapter:
    @staticmethod
    def _layer_and_params(case: DenseCase):
        import jax
        import jax.numpy as jnp

        from nmn.linen import YatNMN

        layer = YatNMN(
            features=case.kernel.shape[1],
            use_bias=True,
            use_alpha=True,
            epsilon=float(case.epsilon),
            learnable_epsilon=True,
            param_dtype=jnp.float32,
        )
        inputs = jnp.asarray(case.inputs, dtype=jnp.float32)
        params = dict(layer.init(jax.random.key(0), inputs)["params"])
        params["kernel"] = jnp.asarray(case.kernel.T, dtype=jnp.float32)
        params["bias"] = jnp.asarray(case.bias, dtype=jnp.float32)
        params["alpha"] = jnp.asarray([float(case.alpha)], dtype=jnp.float32)
        return layer, params

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("flax") is not None

    @staticmethod
    def dense(case: DenseCase, *, compiled: bool = False) -> np.ndarray:
        import jax
        import jax.numpy as jnp

        layer, params = LinenAdapter._layer_and_params(case)
        inputs = jnp.asarray(case.inputs, dtype=jnp.float32)

        def function(value):
            return layer.apply({"params": params}, value)

        if compiled:
            function = jax.jit(function)
        return np.asarray(function(inputs))

    @staticmethod
    def dense_value_and_grad(
        case: DenseCase, *, compiled: bool = False
    ) -> OracleResult:
        import jax
        import jax.numpy as jnp

        layer, params = LinenAdapter._layer_and_params(case)
        inputs = jnp.asarray(case.inputs, dtype=jnp.float32)
        cotangent = jnp.asarray(case.cotangent, dtype=jnp.float32)

        def loss(param_values, input_values):
            output = layer.apply({"params": param_values}, input_values)
            return jnp.sum(output * cotangent), output

        function = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
        if compiled:
            function = jax.jit(function)
        (_, output), (parameter_grads, input_grad) = function(params, inputs)
        raw_scale = jax.nn.sigmoid(params["epsilon_param"])
        gradients = {
            "input": input_grad,
            "kernel": parameter_grads["kernel"].T,
            "bias": parameter_grads["bias"],
            "alpha": jnp.squeeze(parameter_grads["alpha"]),
            "epsilon": jnp.squeeze(parameter_grads["epsilon_param"] / raw_scale),
        }
        return OracleResult(
            np.asarray(output),
            {name: np.asarray(value) for name, value in gradients.items()},
        )
