from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class ReliabilityOutput:
    confidence: float
    signals: Dict[str, Any]
    notes: str


def _energy_concentration_ratio(heatmap: np.ndarray, eps: float = 1e-9) -> float:
    """Top-1% energy / total energy. Max/mean saturates due to heavy-tailed heatmaps; top-k energy ratio is more stable."""
    flat = np.asarray(heatmap).flatten().astype(np.float64)
    flat = np.maximum(flat, 0.0)
    total = flat.sum()
    if total <= 0:
        return 0.0
    n = flat.size
    k = max(1, int(0.01 * n))
    topk = np.partition(flat, -k)[-k:]
    concentration = float(topk.sum()) / (total + eps)
    return float(np.clip(concentration, 0.0, 1.0))


def raw_heatmap_concentration(heatmap: np.ndarray) -> float:
    """Raw energy-based concentration (top-1% / total). Used for calibration reference collection."""
    return _energy_concentration_ratio(heatmap)


def calibrate_concentration(raw: float, calib: Optional[Dict[str, Any]]) -> float:
    """
    Quantile min-max scaling: (raw - c_lo) / (c_hi - c_lo) then clip to [0, 1].
    If calib is None, return raw unchanged.
    Raw energy concentration values are calibrated using a normal reference set to ensure stable confidence scaling.
    Quantile range widened (5–95) to reduce saturation; raw values are tightly distributed.
    """
    if calib is None:
        return raw
    c_lo = float(calib.get("c_lo", 0.0))
    c_hi = float(calib.get("c_hi", 1.0))
    eps = 1e-9
    scaled = (raw - c_lo) / (c_hi - c_lo + eps)
    return float(np.clip(scaled, 0.0, 1.0))


def compute_reliability(
    *,
    score: float,
    threshold: float,
    heatmap_stats: Dict[str, Any],
    heatmap: Optional[np.ndarray] = None,
    concentration_calib: Optional[Dict[str, Any]] = None,
) -> ReliabilityOutput:
    """
    Reliability v1 (rule-based):
    - Higher if score is far from threshold (large margin)
    - Higher if anomaly is localized but salient (reasonable area_ratio + concentration)
    - Lower if score is close to threshold or heatmap is too diffuse / too tiny

    heatmap_stats expected keys:
      - max_intensity (0~1)
      - mean_intensity (0~1)
      - area_ratio (0~1)
    heatmap: optional raw anomaly map for energy-based concentration (top-1% energy ratio).
    concentration_calib: optional dict with c_lo, c_hi from normal reference (quantile min-max). If None, raw concentration is used.
    """
    max_intensity = float(heatmap_stats.get("max_intensity", 0.0))
    mean_intensity = float(heatmap_stats.get("mean_intensity", 0.0))
    area_ratio = float(heatmap_stats.get("area_ratio", 0.0))

    score_margin = abs(float(score) - float(threshold))

    eps = 1e-6
    if heatmap is not None:
        raw_concentration = _energy_concentration_ratio(heatmap, eps=eps)
        # Raw energy concentration values are calibrated using a normal reference set to ensure stable confidence scaling.
        concentration = calibrate_concentration(raw_concentration, concentration_calib)
    else:
        concentration = 0.5

    # area preference: mid-small is often more trustworthy than extremely tiny or huge
    # peak around ~0.03 (3%), penalize extremes
    # simple piecewise scoring
    if area_ratio <= 0.002:
        area_score = 0.2
    elif area_ratio <= 0.08:
        area_score = 1.0
    else:
        area_score = 0.5

    # combine
    # margin (0~1-ish): assume margin above 0.6 is "very confident"
    margin_score = min(1.0, score_margin / 0.6)

    confidence = 0.45 * margin_score + 0.35 * concentration + 0.20 * area_score
    confidence = max(0.0, min(1.0, confidence))

    notes_parts = []
    if score_margin < 0.05:
        notes_parts.append("score close to threshold")
    if concentration < 0.3:
        notes_parts.append("heatmap diffuse")
    if area_ratio <= 0.002:
        notes_parts.append("anomaly area very small")
    elif area_ratio > 0.08:
        notes_parts.append("anomaly area very large")

    notes = "; ".join(notes_parts) if notes_parts else "reliability looks stable"

    return ReliabilityOutput(
        confidence=confidence,
        signals={
            "score_margin": score_margin,
            "heatmap_concentration": concentration,
            "area_ratio": area_ratio,
            "max_intensity": max_intensity,
            "mean_intensity": mean_intensity,
        },
        notes=notes,
    )