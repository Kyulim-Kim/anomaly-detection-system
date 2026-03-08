"""
Threshold policy: resolve a single float threshold from mode and optional inputs.
Prediction uses label = (score >= threshold) only.
"""
from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np

ThresholdMode = Literal["fixed", "normal_p995"]

DEFAULT_FIXED_THRESHOLD = 0.5
NORMAL_P995_QUANTILE = 0.995


def resolve_threshold(
    mode: ThresholdMode = "fixed",
    *,
    fixed_value: float = DEFAULT_FIXED_THRESHOLD,
    normal_scores: Optional[Sequence[float]] = None,
    quantile: float = NORMAL_P995_QUANTILE,
) -> float:
    """
    Compute the anomaly score threshold from policy.

    - "fixed": return fixed_value (default 0.5).
    - "normal_p995": return quantile (default 99.5%) of normal_scores.
      If normal_scores is empty or None, falls back to fixed_value.

    Returns
    -------
    float
        Threshold in [0, 1] for label = (score >= threshold) -> anomaly.
    """
    if mode == "fixed":
        return float(np.clip(fixed_value, 0.0, 1.0))

    if mode == "normal_p995":
        if not normal_scores or len(normal_scores) == 0:
            return float(np.clip(fixed_value, 0.0, 1.0))
        arr = np.asarray(normal_scores, dtype=np.float64)
        q = float(np.nanquantile(arr, quantile))
        return float(np.clip(q, 0.0, 1.0))

    raise ValueError(f"Unknown threshold_mode: {mode!r}")
