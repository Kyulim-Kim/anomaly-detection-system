"""
Uncertain triage: override binary (normal/anomaly) with "uncertain" when borderline or low-trust.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


# Default heuristic constants (tunable via CLI)
DEFAULT_MARGIN_EPS = 0.03
DEFAULT_CONF_EPS = 0.55
DEFAULT_AREA_HI = 0.12
DEFAULT_CONCENTRATION_LO = 0.25

REASON_BORDERLINE_MARGIN = "borderline_margin"
REASON_LOW_CONFIDENCE = "low_confidence"
REASON_TOO_LARGE_AREA = "too_large_area"
REASON_DIFFUSE_HEATMAP = "diffuse_heatmap"


def decide_final_label(
    score: float,
    threshold: float,
    reliability_confidence: float,
    reliability_signals: Dict[str, Any],
    xai_hotspots: List[Any],
    *,
    margin_eps: float = DEFAULT_MARGIN_EPS,
    conf_eps: float = DEFAULT_CONF_EPS,
    area_hi: float = DEFAULT_AREA_HI,
    concentration_lo: float = DEFAULT_CONCENTRATION_LO,
) -> Tuple[str, List[str]]:
    """
    Compute final_label from base (score vs threshold) and optional "uncertain" override.

    Returns
    -------
    final_label : "normal" | "anomaly" | "uncertain"
    reasons : list of human-readable tags for triggered conditions (empty if not uncertain)
    """
    margin = abs(float(score) - float(threshold))
    base_label = "anomaly" if score >= threshold else "normal"

    area_ratio = float(reliability_signals.get("area_ratio", 0.0))
    heatmap_concentration = float(reliability_signals.get("heatmap_concentration", 0.0))

    reasons: List[str] = []
    if margin < margin_eps:
        reasons.append(REASON_BORDERLINE_MARGIN)
    if reliability_confidence < conf_eps:
        reasons.append(REASON_LOW_CONFIDENCE)
    if area_ratio > area_hi:
        reasons.append(REASON_TOO_LARGE_AREA)
    if heatmap_concentration < concentration_lo:
        reasons.append(REASON_DIFFUSE_HEATMAP)

    if reasons:
        final_label = "uncertain"
    else:
        final_label = base_label

    return final_label, reasons
