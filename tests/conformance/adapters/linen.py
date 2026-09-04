"""Flax Linen dense conformance adapter."""

from __future__ import annotations

import importlib.util

import numpy as np

from tests.conformance.oracle import (
    AttentionCase,
    AttentionResult,
    ConvolutionCase,
    DenseCase,
    EmbeddingAttendCase,
    EmbeddingCase,
    OracleResult,
)


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

    @staticmethod
    def embedding_value_and_grad(
        case: EmbeddingCase, *, compiled: bool = False
    ) -> OracleResult:
        import jax
        import jax.numpy as jnp

        from nmn.linen import YatEmbed

        layer = YatEmbed(
            num_embeddings=case.embedding.shape[0],
            features=case.embedding.shape[1],
            param_dtype=jnp.float32,
        )
        indices = jnp.asarray(case.indices, dtype=jnp.int32)
        cotangent = jnp.asarray(case.cotangent, dtype=jnp.float32)
        params = dict(layer.init(jax.random.key(0), indices)["params"])
        params["embedding"] = jnp.asarray(case.embedding, dtype=jnp.float32)

        def loss(parameter_values):
            output = layer.apply({"params": parameter_values}, indices)
            return jnp.sum(output * cotangent), output

        function = jax.value_and_grad(loss, has_aux=True)
        if compiled:
            function = jax.jit(function)
        (_, output), parameter_grads = function(params)
        return OracleResult(
            np.asarray(output), {"embedding": np.asarray(parameter_grads["embedding"])}
        )

    @staticmethod
    def embedding_attend_value_and_grad(
        case: EmbeddingAttendCase, *, compiled: bool = False
    ) -> OracleResult:
        import jax
        import jax.numpy as jnp

        from nmn.linen import YatEmbed

        layer = YatEmbed(
            num_embeddings=case.embedding.shape[0],
            features=case.embedding.shape[1],
            epsilon=float(case.epsilon),
            param_dtype=jnp.float32,
        )
        query = jnp.asarray(case.query, dtype=jnp.float32)
        cotangent = jnp.asarray(case.cotangent, dtype=jnp.float32)
        params = dict(
            layer.init(jax.random.key(0), query, method=layer.attend)["params"]
        )
        params["embedding"] = jnp.asarray(case.embedding, dtype=jnp.float32)
        params["alpha"] = jnp.asarray([float(case.alpha)], dtype=jnp.float32)

        def loss(parameter_values, values):
            output = layer.apply(
                {"params": parameter_values}, values, method=layer.attend
            )
            return jnp.sum(output * cotangent), output

        function = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
        if compiled:
            function = jax.jit(function)
        (_, output), (parameter_grads, query_gradient) = function(params, query)
        return OracleResult(
            np.asarray(output),
            {
                "query": np.asarray(query_gradient),
                "embedding": np.asarray(parameter_grads["embedding"]),
                "alpha": np.asarray(jnp.squeeze(parameter_grads["alpha"])),
            },
        )

    @staticmethod
    def convolution_value_and_grad(
        case: ConvolutionCase, *, transpose: bool, compiled: bool = False
    ) -> OracleResult:
        import jax
        import jax.numpy as jnp

        from nmn.linen import YatConv1D, YatConvTranspose1D

        layer_type = YatConvTranspose1D if transpose else YatConv1D
        layer = layer_type(
            features=case.kernel.shape[-1],
            kernel_size=(case.kernel.shape[0],),
            strides=(1,),
            padding="VALID",
            use_bias=True,
            use_alpha=True,
            epsilon=float(case.epsilon),
            learnable_epsilon=True,
            param_dtype=jnp.float32,
        )
        inputs = jnp.asarray(case.inputs, dtype=jnp.float32)
        cotangent = jnp.asarray(case.cotangent, dtype=jnp.float32)
        params = dict(layer.init(jax.random.key(0), inputs)["params"])
        params["kernel"] = jnp.asarray(case.kernel, dtype=jnp.float32)
        params["bias"] = jnp.asarray(case.bias, dtype=jnp.float32)
        params["alpha"] = jnp.asarray([float(case.alpha)], dtype=jnp.float32)

        def loss(parameter_values, values):
            output = layer.apply({"params": parameter_values}, values)
            return jnp.sum(output * cotangent), output

        function = jax.value_and_grad(loss, argnums=(0, 1), has_aux=True)
        if compiled:
            function = jax.jit(function)
        (_, output), (parameter_grads, input_gradient) = function(params, inputs)
        raw_scale = jax.nn.sigmoid(params["epsilon_param"])
        gradients = {
            "input": input_gradient,
            "kernel": parameter_grads["kernel"],
            "bias": parameter_grads["bias"],
            "alpha": jnp.squeeze(parameter_grads["alpha"]),
            "epsilon": jnp.squeeze(parameter_grads["epsilon_param"] / raw_scale),
        }
        return OracleResult(
            np.asarray(output),
            {name: np.asarray(value) for name, value in gradients.items()},
        )

    @staticmethod
    def attention_value_and_grad(
        case: AttentionCase, *, compiled: bool = False
    ) -> AttentionResult:
        import jax
        import jax.numpy as jnp

        from nmn.linen import yat_attention, yat_attention_weights

        mask = jnp.asarray(case.mask, dtype=bool)
        cotangent = jnp.asarray(case.cotangent, dtype=jnp.float32)

        def loss(query, key, value, alpha, epsilon):
            weights = yat_attention_weights(
                query, key, mask=mask, epsilon=epsilon, alpha=alpha
            )
            output = yat_attention(
                query, key, value, mask=mask, epsilon=epsilon, alpha=alpha
            )
            return jnp.sum(output * cotangent), (weights, output)

        function = jax.value_and_grad(loss, argnums=(0, 1, 2, 3, 4), has_aux=True)
        if compiled:
            function = jax.jit(function)
        operands = (
            jnp.asarray(case.query, dtype=jnp.float32),
            jnp.asarray(case.key, dtype=jnp.float32),
            jnp.asarray(case.value, dtype=jnp.float32),
            jnp.asarray(case.alpha, dtype=jnp.float32),
            jnp.asarray(case.epsilon, dtype=jnp.float32),
        )
        (_, (weights, output)), values = function(*operands)
        gradients = dict(zip(("query", "key", "value", "alpha", "epsilon"), values))
        return AttentionResult(
            np.asarray(weights),
            np.asarray(output),
            {name: np.asarray(item) for name, item in gradients.items()},
        )
