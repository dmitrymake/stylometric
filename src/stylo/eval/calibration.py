"""Compatibility alias for the model-layer calibration implementation.

New model code must import :mod:`stylo.models.calibration`.
"""
from __future__ import annotations

import sys

from ..models import calibration as _canonical

sys.modules[__name__] = _canonical
