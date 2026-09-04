"""Keras dense conformance adapter."""

from __future__ import annotations

import importlib.util

import numpy as np

from tests.conformance.oracle import DenseCase, OracleResult


class KerasAdapter:
    @staticmethod
    def _layer(case: DenseCase):
        import keras

        from nmn.keras import YatNMN

        layer = YatNMN(
            units=case.kernel.shape[1],
            use_bias=True,
            use_alpha=True,
            epsilon=float(case.epsilon),
            learnable_epsilon=True,
            dtype="float32",
        )
        inputs = keras.ops.convert_to_tensor(case.inputs, dtype="float32")
        layer(inputs)
        layer.kernel.assign(case.kernel)
        layer.bias.assign(case.bias)
        layer.alpha.assign([float(case.alpha)])
        return layer

    @staticmethod
    def available() -> bool:
        if importlib.util.find_spec("keras") is None:
            return False
        try:
            import keras  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            return False
        return True

    @staticmethod
    def dense(case: DenseCase, *, compiled: bool = False) -> np.ndarray:
        import keras

        layer = KerasAdapter._layer(case)
        inputs = keras.ops.convert_to_tensor(case.inputs, dtype="float32")
        if compiled and keras.backend.backend() == "tensorflow":
            import tensorflow as tf

            output = tf.function(layer)(inputs)
        elif compiled:
            output = keras.ops.convert_to_numpy(layer(inputs))
        else:
            output = layer(inputs)
        return np.asarray(output)

    @staticmethod
    def dense_value_and_grad(
        case: DenseCase, *, compiled: bool = False
    ) -> OracleResult:
        import keras

        if keras.backend.backend() != "tensorflow":
            raise RuntimeError("Keras conformance gradients require TensorFlow backend")
        import tensorflow as tf

        layer = KerasAdapter._layer(case)
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
        raw_scale = tf.math.sigmoid(layer.epsilon_param.value)
        gradients = {
            "input": values[0],
            "kernel": values[1],
            "bias": values[2],
            "alpha": tf.squeeze(values[3]),
            "epsilon": tf.squeeze(values[4] / raw_scale),
        }
        return OracleResult(
            np.asarray(output),
            {name: np.asarray(value) for name, value in gradients.items()},
        )
