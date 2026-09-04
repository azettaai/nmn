"""PyTorch dense conformance adapter."""

from __future__ import annotations

import importlib.util

import numpy as np

from tests.conformance.oracle import DenseCase, OracleResult


class TorchAdapter:
    @staticmethod
    def _layer(case: DenseCase):
        import torch

        from nmn.torch import YatNMN

        layer = YatNMN(
            case.kernel.shape[0],
            case.kernel.shape[1],
            bias=True,
            alpha=True,
            epsilon=float(case.epsilon),
            learnable_epsilon=True,
            param_dtype=torch.float32,
        )
        with torch.no_grad():
            layer.weight.copy_(torch.asarray(case.kernel.T, dtype=torch.float32))
            layer.bias.copy_(torch.asarray(case.bias, dtype=torch.float32))
            layer.alpha.copy_(torch.asarray([float(case.alpha)], dtype=torch.float32))
        return layer

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("torch") is not None

    @staticmethod
    def dense(case: DenseCase, *, compiled: bool = False) -> np.ndarray:
        import torch

        layer = TorchAdapter._layer(case)
        function = torch.compile(layer) if compiled else layer
        with torch.no_grad():
            output = function(torch.asarray(case.inputs, dtype=torch.float32))
        return output.detach().cpu().numpy()

    @staticmethod
    def dense_value_and_grad(
        case: DenseCase, *, compiled: bool = False
    ) -> OracleResult:
        import torch

        layer = TorchAdapter._layer(case)
        inputs = torch.asarray(case.inputs, dtype=torch.float32).requires_grad_(True)
        function = torch.compile(layer) if compiled else layer
        output = function(inputs)
        loss = torch.sum(output * torch.asarray(case.cotangent, dtype=torch.float32))
        loss.backward()
        raw_scale = torch.sigmoid(layer.epsilon_param.detach())
        gradients = {
            "input": inputs.grad,
            "kernel": layer.weight.grad.T,
            "bias": layer.bias.grad,
            "alpha": layer.alpha.grad.squeeze(),
            "epsilon": (layer.epsilon_param.grad / raw_scale).squeeze(),
        }
        return OracleResult(
            output.detach().cpu().numpy(),
            {name: value.detach().cpu().numpy() for name, value in gradients.items()},
        )
