"""Regression tests for PyTorch dtype and shared-bank issue tranches."""

import copy
import threading

import pytest
import torch

from nmn.torch import (
    MultiHeadYatAttention,
    YatConv1D,
    YatConv2D,
    YatConv3D,
    YatConvTranspose1D,
    YatConvTranspose2D,
    YatConvTranspose3D,
    YatEmbed,
    YatNMN,
)
from nmn.torch._precision import saturating_upcast
