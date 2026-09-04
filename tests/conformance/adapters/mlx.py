"""MLX dense conformance adapter."""

from __future__ import annotations

import numpy as np

from tests._isolated_backend import mlx_is_usable
from tests.conformance.oracle import DenseCase, OracleResult


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
