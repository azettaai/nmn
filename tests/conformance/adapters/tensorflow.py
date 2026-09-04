"""TensorFlow dense conformance adapter."""

from __future__ import annotations

import importlib.util

import numpy as np

from tests.conformance.oracle import DenseCase, OracleResult


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
