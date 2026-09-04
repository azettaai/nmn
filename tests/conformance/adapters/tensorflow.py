"""TensorFlow dense conformance adapter."""

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


class TensorFlowAdapter:
    @staticmethod
    def _layer(case: DenseCase):
        import tensorflow as tf

        from nmn.tf import YatNMN

        layer = YatNMN(
            features=case.kernel.shape[1],
            use_bias=True,
            use_alpha=True,
            epsilon=float(case.epsilon),
            learnable_epsilon=True,
            dtype=tf.float32,
        )
        layer.build(tf.TensorShape(case.inputs.shape))
        layer.kernel.assign(case.kernel.T)
        layer.bias.assign(case.bias)
        layer.alpha.assign([float(case.alpha)])
        return layer

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("tensorflow") is not None

    @staticmethod
    def dense(case: DenseCase, *, compiled: bool = False) -> np.ndarray:
        import tensorflow as tf

        layer = TensorFlowAdapter._layer(case)
        inputs = tf.convert_to_tensor(case.inputs, dtype=tf.float32)
        function = tf.function(layer) if compiled else layer
        return np.asarray(function(inputs))

    @staticmethod
    def dense_value_and_grad(
        case: DenseCase, *, compiled: bool = False
    ) -> OracleResult:
        import tensorflow as tf

        layer = TensorFlowAdapter._layer(case)
        inputs = tf.convert_to_tensor(case.inputs, dtype=tf.float32)
        cotangent = tf.convert_to_tensor(case.cotangent, dtype=tf.float32)

        def evaluate(values):
            with tf.GradientTape() as tape:
                tape.watch(values)
                output = layer(values)
                loss = tf.reduce_sum(output * cotangent)
            gradients = tape.gradient(
                loss,
                [values, layer.kernel, layer.bias, layer.alpha, layer.epsilon_param],
            )
            return output, gradients

        function = tf.function(evaluate) if compiled else evaluate
        output, values = function(inputs)
        raw_scale = tf.math.sigmoid(layer.epsilon_param)
        gradients = {
            "input": values[0],
            "kernel": tf.transpose(values[1]),
            "bias": values[2],
            "alpha": tf.squeeze(values[3]),
            "epsilon": tf.squeeze(values[4] / raw_scale),
        }
        return OracleResult(
            np.asarray(output),
            {name: np.asarray(value) for name, value in gradients.items()},
        )

    @staticmethod
    def embedding_value_and_grad(
        case: EmbeddingCase, *, compiled: bool = False
    ) -> OracleResult:
        import tensorflow as tf

        from nmn.tf import YatEmbed

        layer = YatEmbed(case.embedding.shape[0], case.embedding.shape[1])
        layer.embedding.assign(case.embedding)
        indices = tf.convert_to_tensor(case.indices, dtype=tf.int32)
        cotangent = tf.convert_to_tensor(case.cotangent, dtype=tf.float32)

        def evaluate(values):
            with tf.GradientTape() as tape:
                output = layer(values)
                loss = tf.reduce_sum(output * cotangent)
            return output, tape.gradient(loss, layer.embedding)

        output, gradient = (tf.function(evaluate) if compiled else evaluate)(indices)
        return OracleResult(np.asarray(output), {"embedding": np.asarray(gradient)})

    @staticmethod
    def embedding_attend_value_and_grad(
        case: EmbeddingAttendCase, *, compiled: bool = False
    ) -> OracleResult:
        import tensorflow as tf

        from nmn.tf import YatEmbed

        layer = YatEmbed(
            case.embedding.shape[0],
            case.embedding.shape[1],
            epsilon=float(case.epsilon),
        )
        layer.embedding.assign(case.embedding)
        layer.alpha.assign([float(case.alpha)])
        query = tf.convert_to_tensor(case.query, dtype=tf.float32)
        cotangent = tf.convert_to_tensor(case.cotangent, dtype=tf.float32)

        def evaluate(values):
            with tf.GradientTape() as tape:
                tape.watch(values)
                output = layer.attend(values)
                loss = tf.reduce_sum(output * cotangent)
            gradients = tape.gradient(loss, (values, layer.embedding, layer.alpha))
            return output, gradients

        output, values = (tf.function(evaluate) if compiled else evaluate)(query)
        gradients = dict(zip(("query", "embedding", "alpha"), values))
        gradients["alpha"] = tf.squeeze(gradients["alpha"])
        return OracleResult(
            np.asarray(output),
            {name: np.asarray(value) for name, value in gradients.items()},
        )

    @staticmethod
    def convolution_value_and_grad(
        case: ConvolutionCase, *, transpose: bool, compiled: bool = False
    ) -> OracleResult:
        import tensorflow as tf

        from nmn.tf import YatConv1D, YatConvTranspose1D

        layer_type = YatConvTranspose1D if transpose else YatConv1D
        layer = layer_type(
            filters=case.kernel.shape[-1],
            kernel_size=case.kernel.shape[0],
            padding="valid",
            use_bias=True,
            use_alpha=True,
            epsilon=float(case.epsilon),
            learnable_epsilon=True,
            dtype=tf.float32,
        )
        inputs = tf.convert_to_tensor(case.inputs, dtype=tf.float32)
        layer.build(inputs.shape)
        kernel = case.kernel[::-1].transpose(0, 2, 1) if transpose else case.kernel
        layer.kernel.assign(kernel)
        layer.bias.assign(case.bias)
        layer.alpha.assign([float(case.alpha)])
        cotangent = tf.convert_to_tensor(case.cotangent, dtype=tf.float32)

        def evaluate(values):
            with tf.GradientTape() as tape:
                tape.watch(values)
                output = layer(values)
                loss = tf.reduce_sum(output * cotangent)
            gradients = tape.gradient(
                loss,
                (values, layer.kernel, layer.bias, layer.alpha, layer.epsilon_param),
            )
            return output, gradients

        output, values = (tf.function(evaluate) if compiled else evaluate)(inputs)
        raw_scale = tf.math.sigmoid(layer.epsilon_param)
        kernel_gradient = values[1]
        if transpose:
            kernel_gradient = tf.transpose(kernel_gradient, (0, 2, 1))[::-1]
        gradients = {
            "input": values[0],
            "kernel": kernel_gradient,
            "bias": values[2],
            "alpha": tf.squeeze(values[3]),
            "epsilon": tf.squeeze(values[4] / raw_scale),
        }
        return OracleResult(
            np.asarray(output),
            {name: np.asarray(value) for name, value in gradients.items()},
        )

    @staticmethod
    def attention_value_and_grad(
        case: AttentionCase, *, compiled: bool = False
    ) -> AttentionResult:
        import tensorflow as tf

        from nmn.tf import yat_attention, yat_attention_weights

        mask = tf.convert_to_tensor(case.mask, dtype=tf.bool)
        cotangent = tf.convert_to_tensor(case.cotangent, dtype=tf.float32)

        def evaluate(query, key, value, alpha, epsilon):
            with tf.GradientTape() as tape:
                tape.watch((query, key, value, alpha, epsilon))
                weights = yat_attention_weights(
                    query, key, mask=mask, epsilon=epsilon, alpha=alpha
                )
                output = yat_attention(
                    query, key, value, mask=mask, epsilon=epsilon, alpha=alpha
                )
                loss = tf.reduce_sum(output * cotangent)
            gradients = tape.gradient(loss, (query, key, value, alpha, epsilon))
            return weights, output, gradients

        function = tf.function(evaluate) if compiled else evaluate
        operands = (
            tf.convert_to_tensor(case.query, dtype=tf.float32),
            tf.convert_to_tensor(case.key, dtype=tf.float32),
            tf.convert_to_tensor(case.value, dtype=tf.float32),
            tf.convert_to_tensor(case.alpha, dtype=tf.float32),
            tf.convert_to_tensor(case.epsilon, dtype=tf.float32),
        )
        weights, output, values = function(*operands)
        gradients = dict(zip(("query", "key", "value", "alpha", "epsilon"), values))
        return AttentionResult(
            np.asarray(weights),
            np.asarray(output),
            {name: np.asarray(item) for name, item in gradients.items()},
        )
