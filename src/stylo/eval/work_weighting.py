"""Compatibility alias for the evaluation-independent weighting contract.

New implementation code must import :mod:`stylo.domain.work_weighting`.
"""
from __future__ import annotations

import sys

from ..domain import work_weighting as _canonical

sys.modules[__name__] = _canonical
