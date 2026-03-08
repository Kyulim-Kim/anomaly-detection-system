"""Unit tests for threshold resolution (no model inference)."""
from __future__ import annotations

import pytest

from src.pipeline.threshold import resolve_threshold
import numpy as np


def test_fixed_mode_returns_provided_threshold() -> None:
    """Fixed mode returns the given value, clipped to [0, 1]."""
    assert resolve_threshold("fixed", fixed_value=0.5) == 0.5
    assert resolve_threshold("fixed", fixed_value=0.0) == 0.0
    assert resolve_threshold("fixed", fixed_value=1.0) == 1.0
    assert resolve_threshold("fixed", fixed_value=0.7) == 0.7
    # clipping
    assert resolve_threshold("fixed", fixed_value=1.5) == 1.0
    assert resolve_threshold("fixed", fixed_value=-0.1) == 0.0


def test_normal_p995_returns_float_from_scores() -> None:
    scores = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    t = resolve_threshold("normal_p995", normal_scores=scores, quantile=0.995)
    expected = float(np.quantile(scores, 0.995))

    assert isinstance(t, float)
    assert t == expected

def test_normal_p995_empty_falls_back_to_fixed() -> None:
    """normal_p995 with empty or None normal_scores uses fixed_value."""
    assert resolve_threshold("normal_p995", normal_scores=[], fixed_value=0.5) == 0.5
    assert resolve_threshold("normal_p995", normal_scores=None, fixed_value=0.3) == 0.3


def test_unknown_mode_raises() -> None:
    """Unknown threshold mode raises ValueError."""
    with pytest.raises(ValueError, match="Unknown threshold_mode"):
        resolve_threshold("invalid_mode")
    with pytest.raises(ValueError, match="unknown"):
        resolve_threshold("unknown")
