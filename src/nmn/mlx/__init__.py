"""MLX backend for Neural Matter Network (NMN).

Apple-Silicon-native implementation of the YAT family of layers, mirroring
the surface of ``nmn.tf`` / ``nmn.keras``. Requires ``mlx``.
"""

from .nmn import YatNMN, YatDense
from .squashers import softermax, softer_sigmoid, soft_tanh

__all__ = [
    "YatNMN",
    "YatDense",
    "softermax",
    "softer_sigmoid",
    "soft_tanh",
]
