from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Evaluation:
    has_gt: bool
    gt_label: Optional[str]
    error_type: Optional[str]


def _normalize_pred(label: Optional[str]) -> Optional[str]:
    """Normalize prediction label to normal | anomaly | uncertain."""
    if label is None:
        return None
    s = label.strip().lower()
    if s == "normal":
        return "normal"
    if s == "anomaly":
        return "anomaly"
    if s == "uncertain":
        return "uncertain"
    return None


def _gt_display_to_normal_anomaly(gt_label_display: Optional[str]) -> Optional[str]:
    """
    Map display GT label to normal/anomaly for TP/FP/TN/FN.
    - "good" or "normal" -> "normal"
    - "anomaly" or any defect_type (e.g. "broken_small") -> "anomaly"
    """
    if gt_label_display is None:
        return None
    s = str(gt_label_display).strip().lower()
    if s in {"good", "normal"}:
        return "normal"
    if s == "anomaly":
        return "anomaly"
    return "anomaly"


def compute_evaluation(
    pred_label: Optional[str],
    gt_label_display: Optional[str],
    has_gt: bool,
) -> Dict[str, Any]:
    """
    Compute evaluation fields: has_gt, gt_label (display), error_type.

    - gt_label_display: "good" | defect_type (e.g. "broken_small") | "normal"|"anomaly" for backward compat.
    - has_gt: True when a GT mask file was found and used.
    - error_type: TP/FP/TN/FN only when has_gt and pred is not "uncertain"; else null.
    """
    n_pred = _normalize_pred(pred_label)
    n_gt = _gt_display_to_normal_anomaly(gt_label_display)

    if not has_gt:
        return asdict(Evaluation(has_gt=False, gt_label=gt_label_display, error_type=None))

    if n_pred not in {"normal", "anomaly"}:
        return asdict(Evaluation(has_gt=True, gt_label=gt_label_display, error_type=None))

    if n_gt not in {"normal", "anomaly"}:
        return asdict(Evaluation(has_gt=True, gt_label=gt_label_display, error_type=None))

    if n_gt == "anomaly" and n_pred == "anomaly":
        et = "TP"
    elif n_gt == "normal" and n_pred == "anomaly":
        et = "FP"
    elif n_gt == "normal" and n_pred == "normal":
        et = "TN"
    elif n_gt == "anomaly" and n_pred == "normal":
        et = "FN"
    else:
        et = None

    return asdict(Evaluation(has_gt=True, gt_label=gt_label_display, error_type=et))

