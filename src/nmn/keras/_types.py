"""Typing boundaries for Keras's runtime-selected tensor and configuration APIs.

Keras ops return TensorFlow, JAX, Torch, or symbolic Keras tensors depending on
runtime backend/tracing. Keras does not expose one static Tensor base class.
Any is restricted to that tensor boundary and extensible serialized configs;
scalar options, shapes, return structures, and optionality remain explicit.
"""

from typing import Any, TypeAlias

Tensor: TypeAlias = Any
Shape: TypeAlias = tuple[int | None, ...] | list[int | None]
Config: TypeAlias = dict[str, Any]
