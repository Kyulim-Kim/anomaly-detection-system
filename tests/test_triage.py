"""Unit tests for triage (decide_final_label)."""
from __future__ import annotations

from src.pipeline.triage import (
    REASON_BORDERLINE_MARGIN,
    REASON_DIFFUSE_HEATMAP,
    REASON_LOW_CONFIDENCE,
    REASON_TOO_LARGE_AREA,
    decide_final_label,
)


def test_base_anomaly_no_uncertain_conditions() -> None:
    """Base anomaly with no triggered uncertain conditions => final anomaly."""
    score = 0.8
    threshold = 0.5
    reliability_confidence = 0.7
    reliability_signals = {"area_ratio": 0.05, "heatmap_concentration": 0.6}
    hotspots = []
    final_label, reasons = decide_final_label(
        score=score,
        threshold=threshold,
        reliability_confidence=reliability_confidence,
        reliability_signals=reliability_signals,
        xai_hotspots=hotspots,
    )
    assert final_label == "anomaly"
    assert reasons == []


def test_score_near_threshold_borderline_margin() -> None:
    """Score near threshold => uncertain with borderline_margin."""
    threshold = 0.5
    score = threshold + 0.02  # margin 0.02 < margin_eps 0.03
    reliability_confidence = 0.8
    reliability_signals = {"area_ratio": 0.05, "heatmap_concentration": 0.6}
    hotspots = []
    final_label, reasons = decide_final_label(
        score=score,
        threshold=threshold,
        reliability_confidence=reliability_confidence,
        reliability_signals=reliability_signals,
        xai_hotspots=hotspots,
        margin_eps=0.03,
    )
    assert final_label == "uncertain"
    assert REASON_BORDERLINE_MARGIN in reasons


def test_low_confidence_uncertain() -> None:
    """Low reliability confidence => uncertain with low_confidence."""
    score = 0.6
    threshold = 0.5
    reliability_confidence = 0.4  # below conf_eps 0.55
    reliability_signals = {"area_ratio": 0.05, "heatmap_concentration": 0.6}
    hotspots = []
    final_label, reasons = decide_final_label(
        score=score,
        threshold=threshold,
        reliability_confidence=reliability_confidence,
        reliability_signals=reliability_signals,
        xai_hotspots=hotspots,
        conf_eps=0.55,
    )
    assert final_label == "uncertain"
    assert REASON_LOW_CONFIDENCE in reasons


def test_large_area_ratio_uncertain() -> None:
    """Large area_ratio => uncertain with too_large_area."""
    score = 0.6
    threshold = 0.5
    reliability_confidence = 0.8
    reliability_signals = {"area_ratio": 0.2, "heatmap_concentration": 0.6}  # 0.2 > area_hi 0.12
    hotspots = []
    final_label, reasons = decide_final_label(
        score=score,
        threshold=threshold,
        reliability_confidence=reliability_confidence,
        reliability_signals=reliability_signals,
        xai_hotspots=hotspots,
        area_hi=0.12,
    )
    assert final_label == "uncertain"
    assert REASON_TOO_LARGE_AREA in reasons


def test_low_concentration_uncertain() -> None:
    """Low heatmap concentration => uncertain with diffuse_heatmap."""
    score = 0.6
    threshold = 0.5
    reliability_confidence = 0.8
    reliability_signals = {"area_ratio": 0.05, "heatmap_concentration": 0.1}  # 0.1 < concentration_lo 0.25
    hotspots = []
    final_label, reasons = decide_final_label(
        score=score,
        threshold=threshold,
        reliability_confidence=reliability_confidence,
        reliability_signals=reliability_signals,
        xai_hotspots=hotspots,
        concentration_lo=0.25,
    )
    assert final_label == "uncertain"
    assert REASON_DIFFUSE_HEATMAP in reasons
