"""Shared implementation for the canonical and legacy import paths."""

import sys

from jev_mlx import token_codes as _implementation

sys.modules[__name__] = _implementation
