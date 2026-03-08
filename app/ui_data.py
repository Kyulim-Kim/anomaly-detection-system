"""Data loading and transformation helpers for the Streamlit viewer (no UI)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent

REASON_ORDER = ["borderline_margin", "low_confidence", "too_large_area", "diffuse_heatmap"]

_LAZY_KEYS = ("label", "score", "confidence", "error_type", "triage_final_label", "triage_reasons", "defect_type", "result_relpath")


def get_project_root() -> Path:
    return _PROJECT_ROOT


def discover_runs(root: str = "artifacts/runs") -> List[Dict[str, str]]:
    """Discover runs by scanning for run_meta.json under root. Returns [{display, run_dir}, ...]."""
    base = _PROJECT_ROOT / root
    if not base.exists():
        return []
    out: List[Dict[str, str]] = []
    for meta_path in sorted(base.rglob("run_meta.json")):
        run_dir_path = meta_path.parent
        run_dir_str = str(run_dir_path)
        display = run_dir_str
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta.get("run_id"), str):
                display = meta["run_id"].strip() or display
        except Exception:
            pass
        if not display or display == run_dir_str:
            try:
                display = str(run_dir_path.relative_to(_PROJECT_ROOT))
            except ValueError:
                display = run_dir_path.name
        out.append({"display": display, "run_dir": run_dir_str})
    return out


def get_final_label(row: Dict[str, Any]) -> str:
    """Final label: triage.final_label if present, else prediction.label."""
    lb = (row.get("label") or row.get("triage_final_label") or "").strip().lower()
    return lb if lb in ("normal", "anomaly", "uncertain") else ""


def get_reasons(row: Dict[str, Any]) -> List[str]:
    """Triage reasons as list (never None)."""
    r = row.get("triage_reasons")
    if r is None:
        return []
    return list(r) if isinstance(r, (list, tuple)) else []


def get_base_label(row: Dict[str, Any], threshold: Optional[float]) -> str:
    """Pre-triage label: anomaly if score >= threshold, else normal. Uses run-level threshold."""
    if threshold is None:
        return "normal"
    score = float(row.get("score") or 0)
    return "anomaly" if score >= threshold else "normal"


def reason_hit_rates(uncertain_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """For each reason in REASON_ORDER, hit_rate = (# samples containing it) / total uncertain. Multi-label."""
    n = len(uncertain_rows)
    if n == 0:
        return {}
    counts = {reason: 0 for reason in REASON_ORDER}
    for row in uncertain_rows:
        reasons = get_reasons(row)
        for reason in REASON_ORDER:
            if reason in reasons:
                counts[reason] += 1
    return {reason: counts[reason] / n for reason in REASON_ORDER}


def drift_hint_text(reason_hit_rates_dict: Dict[str, float]) -> str:
    """One-line drift hint if a reason dominates (>= 0.6)."""
    if reason_hit_rates_dict.get("diffuse_heatmap", 0) >= 0.6:
        return "Drift hint: diffuse_heatmap is frequent — check illumination/reflections/background shift."
    if reason_hit_rates_dict.get("too_large_area", 0) >= 0.6:
        return "Drift hint: too_large_area is frequent — check ROI/background masking and product positioning."
    return ""


def top_uncertain_rows(rows: List[Dict[str, Any]], threshold: Optional[float], n: int = 3) -> List[Dict[str, Any]]:
    """Top N uncertain samples as list of dicts for dataframe (sample_id, filename, score, confidence, margin, reasons)."""
    uncertain = [r for r in rows if get_final_label(r) == "uncertain"]
    if not uncertain:
        return []

    def sort_key(r: Dict[str, Any]) -> tuple:
        sc = float(r.get("score") or 0)
        conf = float(r.get("confidence") or 0)
        margin = abs(sc - threshold) if threshold is not None else 999.0
        return (margin, conf)

    top = sorted(uncertain, key=sort_key)[:n]
    out = []
    for r in top:
        sid = r.get("sample_id") or "—"
        fn = r.get("input_filename") or "—"
        sc = float(r.get("score") or 0)
        conf = float(r.get("confidence") or 0)
        margin = abs(sc - threshold) if threshold is not None else None
        margin_str = f"{margin:.4f}" if margin is not None else "—"
        reas = ", ".join(get_reasons(r)) or "—"
        out.append({"sample_id": sid, "filename": fn, "score": round(sc, 4), "confidence": round(conf, 3), "margin": margin_str, "reasons": reas})
    return out


def load_index(index_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not index_path.exists():
        return rows
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _fill_row_from_result(row: Dict[str, Any], run_dir: Path) -> None:
    """In-place: read sample's result.json and fill missing fields."""
    sample_id = row.get("sample_id")
    if not sample_id:
        return
    relpath = row.get("result_relpath")
    if not relpath:
        relpath = f"samples/{sample_id}/result.json"
        row["result_relpath"] = relpath
    result_path = run_dir / relpath
    if not result_path.exists():
        return
    try:
        with result_path.open("r", encoding="utf-8") as f:
            r = json.load(f)
    except Exception:
        return
    pred = r.get("prediction", {})
    rel = r.get("reliability", {})
    eval_ = r.get("evaluation", {})
    meta = r.get("meta", {})
    triage = r.get("triage", {})
    if row.get("label") is None:
        row["label"] = pred.get("label")
    if row.get("score") is None:
        row["score"] = pred.get("score")
    if row.get("confidence") is None:
        row["confidence"] = rel.get("confidence")
    if row.get("error_type") is None:
        row["error_type"] = eval_.get("error_type")
    if row.get("defect_type") is None:
        row["defect_type"] = meta.get("defect_type")
    if row.get("triage_final_label") is None:
        row["triage_final_label"] = triage.get("final_label")
    if row.get("triage_reasons") is None:
        row["triage_reasons"] = triage.get("reasons")
    if row.get("input_filename") is None:
        row["input_filename"] = meta.get("input_filename")
    if row.get("created_at") is None:
        row["created_at"] = r.get("created_at")


def load_run_results(run_dir: Path) -> List[Dict[str, Any]]:
    """Use index.jsonl as primary; lazy-fill missing fields from each sample's result.json when needed."""
    run_dir = Path(run_dir)
    index_path = run_dir / "index.jsonl"
    if index_path.exists():
        rows = load_index(index_path)
        for row in rows:
            if any(row.get(k) is None for k in _LAZY_KEYS):
                _fill_row_from_result(row, run_dir)
        return rows
    rows = []
    samples_dir = run_dir / "samples"
    if not samples_dir.exists():
        return rows
    for sample_dir in sorted(samples_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        result_path = sample_dir / "result.json"
        if not result_path.exists():
            continue
        try:
            with result_path.open("r", encoding="utf-8") as f:
                r = json.load(f)
        except Exception:
            continue
        pred = r.get("prediction", {})
        rel = r.get("reliability", {})
        eval_ = r.get("evaluation", {})
        meta = r.get("meta", {})
        triage = r.get("triage", {})
        sample_id = r.get("sample_id") or sample_dir.name
        rows.append({
            "sample_id": sample_id,
            "result_relpath": f"samples/{sample_id}/result.json",
            "label": pred.get("label"),
            "score": pred.get("score"),
            "confidence": rel.get("confidence"),
            "error_type": eval_.get("error_type"),
            "triage_final_label": triage.get("final_label"),
            "triage_reasons": triage.get("reasons"),
            "defect_type": meta.get("defect_type"),
            "input_filename": meta.get("input_filename"),
            "created_at": r.get("created_at"),
        })
    return rows


def get_run_threshold(run_dir: Path) -> Optional[float]:
    """Threshold from run_meta.json or first result.json."""
    t, _ = get_run_threshold_info(run_dir)
    return t


def get_run_threshold_info(run_dir: Path) -> tuple[Optional[float], Optional[str]]:
    """Returns (threshold, mode). mode from run_meta threshold_policy.mode else 'unknown'."""
    run_dir = Path(run_dir)
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            tp = meta.get("threshold_policy") or {}
            t = tp.get("threshold")
            mode = tp.get("mode")
            if t is not None:
                return float(t), (mode if isinstance(mode, str) else "unknown")
        except Exception:
            pass
    rows = load_run_results(run_dir)
    for r in rows:
        relpath = r.get("result_relpath")
        if not relpath:
            continue
        try:
            res = load_result(run_dir, relpath)
            t = (res.get("prediction") or {}).get("threshold")
            if t is not None:
                return float(t), "unknown"
        except Exception:
            continue
    return None, None


def load_result(run_dir: Path, result_relpath: str) -> Dict[str, Any]:
    p = run_dir / result_relpath
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_run_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """total_samples, label_counts (normal/anomaly/uncertain), ratios."""
    total = len(rows)
    counts = {"normal": 0, "anomaly": 0, "uncertain": 0}
    for r in rows:
        lb = (r.get("label") or "").strip().lower()
        if lb in counts:
            counts[lb] += 1
    return {
        "total_samples": total,
        "label_counts": counts,
        "uncertain_ratio": counts["uncertain"] / total if total else 0.0,
        "anomaly_ratio": counts["anomaly"] / total if total else 0.0,
        "normal_ratio": counts["normal"] / total if total else 0.0,
    }
