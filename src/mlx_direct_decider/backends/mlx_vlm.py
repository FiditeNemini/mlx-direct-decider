"""Shared implementation for the canonical and legacy import paths."""

import sys

from jev_mlx.backends import mlx_vlm as _implementation

sys.modules[__name__] = _implementation
