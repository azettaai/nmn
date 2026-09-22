"""Private Keras hooks retained for serialized names and legacy shape semantics.

Kept behind one adapter so minimum/current Keras tests exercise the same boundary.
Public Keras layer, ops, and saving APIs must be imported from keras directly.
"""

from keras.src.api_export import keras_export
from keras.src.backend.common.backend_utils import compute_conv_transpose_output_shape
from keras.src.ops.operation_utils import compute_conv_output_shape

__all__ = [
    "keras_export",
    "compute_conv_output_shape",
    "compute_conv_transpose_output_shape",
]
