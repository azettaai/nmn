"""MLX dense conformance adapter."""

from __future__ import annotations

import numpy as np

from tests._isolated_backend import mlx_is_usable
from tests.conformance.oracle import (
    AttentionCase,
    AttentionResult,
    ConvolutionCase,
    DenseCase,
    EmbeddingAttendCase,
    EmbeddingCase,
    OracleResult,
)


class MlxAdapter:
    @staticmethod
    def _layer(case: DenseCase):
        import mlx.core as mx

        from nmn.mlx import YatNMN

        layer = YatNMN(
            features=case.kernel.shape[1],
            use_bias=True,
            use_alpha=True,
            epsilon=float(case.epsilon),
            learnable_epsilon=True,
            param_dtype=mx.float32,
        )
        layer.build(case.kernel.shape[0])
        layer.kernel = mx.array(case.kernel.T, dtype=mx.float32)
        layer.bias = mx.array(case.bias, dtype=mx.float32)
        layer.alpha = mx.array([float(case.alpha)], dtype=mx.float32)
        return layer

    @staticmethod
    def available() -> bool:
        return mlx_is_usable()

    @staticmethod
    def dense(case: DenseCase, *, compiled: bool = False) -> np.ndarray:
        import mlx.core as mx

        layer = MlxAdapter._layer(case)
        function = mx.compile(layer, inputs=layer.state) if compiled else layer
        output = function(mx.array(case.inputs, dtype=mx.float32))
        mx.eval(output)
        return np.asarray(output)

    @staticmethod
    def dense_value_and_grad(
        case: DenseCase, *, compiled: bool = False
    ) -> OracleResult:
        import mlx.core as mx
        import mlx.nn as nn

        layer = MlxAdapter._layer(case)
        inputs = mx.array(case.inputs, dtype=mx.float32)
        cotangent = mx.array(case.cotangent, dtype=mx.float32)

        def loss(model, values):
            output = model(values)
            return mx.sum(output * cotangent), output

        parameter_grad = nn.value_and_grad(layer, loss)
        input_grad = mx.grad(lambda values: loss(layer, values)[0])

        def evaluate(values):
            (value, output), parameter_values = parameter_grad(layer, values)
            del value
            return output, parameter_values, input_grad(values)

        function = mx.compile(evaluate, inputs=layer.state) if compiled else evaluate
        output, parameter_values, input_value = function(inputs)
        raw_scale = mx.sigmoid(layer.epsilon_param)
        gradients = {
            "input": input_value,
            "kernel": mx.transpose(parameter_values["kernel"]),
            "bias": parameter_values["bias"],
            "alpha": mx.squeeze(parameter_values["alpha"]),
            "epsilon": mx.squeeze(parameter_values["epsilon_param"] / raw_scale),
        }
        mx.eval(output, gradients)
        return OracleResult(
            np.asarray(output),
            {name: np.asarray(value) for name, value in gradients.items()},
        )

    @staticmethod
    def embedding_value_and_grad(
        case: EmbeddingCase, *, compiled: bool = False
    ) -> OracleResult:
        import mlx.core as mx
        import mlx.nn as nn

        from nmn.mlx import YatEmbed

        layer = YatEmbed(case.embedding.shape[0], case.embedding.shape[1])
        layer.embedding = mx.array(case.embedding, dtype=mx.float32)
        indices = mx.array(case.indices, dtype=mx.int32)
        cotangent = mx.array(case.cotangent, dtype=mx.float32)
        gradient_fn = nn.value_and_grad(
            layer, lambda model, values: mx.sum(model(values) * cotangent)
        )
        function = (
            mx.compile(lambda values: gradient_fn(layer, values), inputs=layer.state)
            if compiled
            else lambda values: gradient_fn(layer, values)
        )
        _, gradients = function(indices)
        output = layer(indices)
        mx.eval(output, gradients)
        return OracleResult(
            np.asarray(output), {"embedding": np.asarray(gradients["embedding"])}
        )

    @staticmethod
    def embedding_attend_value_and_grad(
        case: EmbeddingAttendCase, *, compiled: bool = False
    ) -> OracleResult:
        import mlx.core as mx
        import mlx.nn as nn

        from nmn.mlx import YatEmbed

        layer = YatEmbed(
            case.embedding.shape[0],
            case.embedding.shape[1],
            epsilon=float(case.epsilon),
        )
        layer.embedding = mx.array(case.embedding, dtype=mx.float32)
        layer.alpha = mx.array([float(case.alpha)], dtype=mx.float32)
        query = mx.array(case.query, dtype=mx.float32)
        cotangent = mx.array(case.cotangent, dtype=mx.float32)
        parameter_fn = nn.value_and_grad(
            layer, lambda model, values: mx.sum(model.attend(values) * cotangent)
        )
        query_fn = mx.value_and_grad(
            lambda values: mx.sum(layer.attend(values) * cotangent)
        )

        def evaluate(values):
            _, parameter_gradients = parameter_fn(layer, values)
            _, query_gradient = query_fn(values)
            return layer.attend(values), parameter_gradients, query_gradient

        function = mx.compile(evaluate, inputs=layer.state) if compiled else evaluate
        output, parameter_gradients, query_gradient = function(query)
        gradients = {
            "query": query_gradient,
            "embedding": parameter_gradients["embedding"],
            "alpha": mx.squeeze(parameter_gradients["alpha"]),
        }
        mx.eval(output, gradients)
        return OracleResult(
            np.asarray(output),
            {name: np.asarray(value) for name, value in gradients.items()},
        )

    @staticmethod
    def convolution_value_and_grad(
        case: ConvolutionCase, *, transpose: bool, compiled: bool = False
    ) -> OracleResult:
        import mlx.core as mx
        import mlx.nn as nn

        from nmn.mlx import YatConv1D, YatConvTranspose1D

        layer_type = YatConvTranspose1D if transpose else YatConv1D
        layer = layer_type(
            filters=case.kernel.shape[-1],
            kernel_size=case.kernel.shape[0],
            padding="valid",
            use_bias=True,
            use_alpha=True,
            epsilon=float(case.epsilon),
            learnable_epsilon=True,
            dtype=mx.float32,
        )
        layer.build(case.inputs.shape[-1])
        kernel = (
            case.kernel[::-1].transpose(2, 0, 1)
            if transpose
            else case.kernel.transpose(2, 0, 1)
        )
        layer.kernel = mx.array(kernel, dtype=mx.float32)
        layer.bias = mx.array(case.bias, dtype=mx.float32)
        layer.alpha = mx.array([float(case.alpha)], dtype=mx.float32)
        inputs = mx.array(case.inputs, dtype=mx.float32)
        cotangent = mx.array(case.cotangent, dtype=mx.float32)
        parameter_fn = nn.value_and_grad(
            layer, lambda model, values: mx.sum(model(values) * cotangent)
        )
        input_fn = mx.value_and_grad(lambda values: mx.sum(layer(values) * cotangent))

        def evaluate(values):
            _, parameter_gradients = parameter_fn(layer, values)
            _, input_gradient = input_fn(values)
            return layer(values), parameter_gradients, input_gradient

        function = mx.compile(evaluate, inputs=layer.state) if compiled else evaluate
        output, parameter_gradients, input_gradient = function(inputs)
        kernel_gradient = mx.transpose(parameter_gradients["kernel"], (1, 2, 0))
        if transpose:
            kernel_gradient = kernel_gradient[::-1]
        raw_scale = mx.sigmoid(layer.epsilon_param)
        gradients = {
            "input": input_gradient,
            "kernel": kernel_gradient,
            "bias": parameter_gradients["bias"],
            "alpha": mx.squeeze(parameter_gradients["alpha"]),
            "epsilon": mx.squeeze(parameter_gradients["epsilon_param"] / raw_scale),
        }
        mx.eval(output, gradients)
        return OracleResult(
            np.asarray(output),
            {name: np.asarray(value) for name, value in gradients.items()},
        )

    @staticmethod
    def attention_value_and_grad(
        case: AttentionCase, *, compiled: bool = False
    ) -> AttentionResult:
        import mlx.core as mx

        from nmn.mlx import yat_attention, yat_attention_weights

        mask = mx.array(case.mask, dtype=mx.bool_)
        cotangent = mx.array(case.cotangent, dtype=mx.float32)

        def loss(query, key, value, alpha, epsilon):
            weights = yat_attention_weights(
                query, key, mask=mask, epsilon=epsilon, alpha=alpha
            )
            output = yat_attention(
                query, key, value, mask=mask, epsilon=epsilon, alpha=alpha
            )
            return mx.sum(output * cotangent), (weights, output)

        function = mx.value_and_grad(loss, argnums=(0, 1, 2, 3, 4), has_aux=True)
        if compiled:
            function = mx.compile(function)
        operands = tuple(
            mx.array(value, dtype=mx.float32)
            for value in (
                case.query,
                case.key,
                case.value,
                case.alpha,
                case.epsilon,
            )
        )
        (_, (weights, output)), values = function(*operands)
        gradients = dict(zip(("query", "key", "value", "alpha", "epsilon"), values))
        mx.eval(weights, output, gradients)
        return AttentionResult(
            np.asarray(weights),
            np.asarray(output),
            {name: np.asarray(item) for name, item in gradients.items()},
        )
