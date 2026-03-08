"""
ECDF-based Risk (est.): Risk = CDF_normal(score).
Risk (est.) increases as the score becomes more anomalous (higher score → higher CDF under normal distribution).
Uses normal decision-score distribution; no extra dependencies (bisect only).
"""
from __future__ import annotations

import bisect
from typing import Dict, List, Optional


def build_ecdf(
    values: List[float],
    *,
    max_points: Optional[int] = 5000,
) -> Dict:
    """
    Build ECDF descriptor from a list of scores (e.g. normal decision scores).

    Returns
    -------
    dict with:
      - method: "ecdf_normal"
      - n: int (original count)
      - scores_sorted: list[float] ascending; downsampled if len(values) > max_points
    """
    if not values:
        return {"method": "ecdf_normal", "n": 0, "scores_sorted": []}
    scores_sorted = sorted(float(v) for v in values)
    n = len(scores_sorted)
    if max_points is not None and n > max_points:
        # Equidistant indices to keep max_points points (include min and max)
        step = (n - 1) / (max_points - 1) if max_points > 1 else 0
        indices = [0] + [int(round(i * step)) for i in range(1, max_points - 1)] + [n - 1]
        scores_sorted = [scores_sorted[i] for i in indices]
    return {
        "method": "ecdf_normal",
        "n": n,
        "scores_sorted": scores_sorted,
    }


def ecdf_risk_prob(ecdf: Dict, score: float) -> float:
    """
    Risk (est.) = CDF(score) = (# of normal scores <= score) / n.
    Risk (est.) increases as the score becomes more anomalous.

    Uses bisect_right on scores_sorted; result clipped to [0, 1].
    """
    scores_sorted = ecdf.get("scores_sorted") or []
    n = ecdf.get("n") or 0
    if n == 0 or not scores_sorted:
        return 0.0
    idx = bisect.bisect_right(scores_sorted, score)
    cdf = idx / n
    return max(0.0, min(1.0, float(cdf)))


def _check_ecdf_sanity() -> None:
    """Quick sanity check: CDF at max score = 1, at min score = 0; higher score → higher risk."""
    vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    ecdf = build_ecdf(vals, max_points=10)
    assert ecdf["n"] == 5 and ecdf["method"] == "ecdf_normal"
    assert ecdf_risk_prob(ecdf, 0.5) == 1.0
    assert ecdf_risk_prob(ecdf, 0.0) == 0.0
    assert 0.0 < ecdf_risk_prob(ecdf, 0.25) < 1.0


if __name__ == "__main__":
    _check_ecdf_sanity()
    print("calibration sanity check OK")
