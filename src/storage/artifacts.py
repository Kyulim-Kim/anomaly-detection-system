from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from src.storage.schema_validation import validate_result


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_png(path: Path, img: Image.Image) -> None:
    img.save(path, format="PNG")


def save_npy(path: Path, arr: np.ndarray) -> None:
    np.save(path, arr)


def init_run_dir(out_root: Path, run_id: str) -> Path:
    run_dir = out_root / run_id
    ensure_dir(run_dir / "samples")
    # index.jsonl is append-only; create if missing
    index_path = run_dir / "index.jsonl"
    if not index_path.exists():
        index_path.write_text("", encoding="utf-8")
    return run_dir


def write_run_meta(run_dir: Path, meta: Dict[str, Any]) -> None:
    write_json(run_dir / "run_meta.json", meta)


def load_run_meta(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load run_meta.json if it exists; otherwise return None."""
    path = run_dir / "run_meta.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_sample_artifacts(
    run_dir: Path,
    sample_id: str,
    *,
    original: Image.Image,
    heatmap: Image.Image,
    overlay: Image.Image,
    result: Dict[str, Any],
    anomaly_map: Optional[np.ndarray] = None,
    gt_mask: Optional[Image.Image] = None,
    validate_schema_path: Optional[Path] = None,
) -> Path:
    """
    Write artifacts under:
      <run_dir>/samples/<sample_id>/

    Constraint:
    - All paths stored in `result["paths"]` must be relative to this sample dir.
    """
    sample_dir = run_dir / "samples" / sample_id
    ensure_dir(sample_dir)
    debug_dir_name = result.get("paths", {}).get("debug_dir")
    if debug_dir_name:
        ensure_dir(sample_dir / debug_dir_name)

    # Schema validation (optional) before saving.
    if validate_schema_path is not None:
        validate_result(result, validate_schema_path)

    # Save images/files
    save_png(sample_dir / "original.png", original)
    save_png(sample_dir / "heatmap.png", heatmap)
    save_png(sample_dir / "overlay.png", overlay)
    if gt_mask is not None:
        save_png(sample_dir / "gt_mask.png", gt_mask)
    if anomaly_map is not None:
        save_npy(sample_dir / "anomaly_map.npy", anomaly_map.astype(np.float32))

    write_json(sample_dir / "result.json", result)
    return sample_dir


def index_record_for_sample(run_id: str, sample_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    `index.jsonl` record. Keep it small; UI can load result.json when needed.
    Paths here are relative to run_dir for easy navigation by Streamlit.
    """
    pred = result.get("prediction", {})
    rel = result.get("reliability", {})
    eval_ = result.get("evaluation", {})
    meta = result.get("meta", {})

    return {
        "run_id": run_id,
        "sample_id": sample_id,
        "result_relpath": f"samples/{sample_id}/result.json",
        "overlay_relpath": f"samples/{sample_id}/overlay.png",
        "original_relpath": f"samples/{sample_id}/original.png",
        "label": pred.get("label"),
        "score": pred.get("score"),
        "confidence": rel.get("confidence"),
        "error_type": eval_.get("error_type"),
        "input_filename": meta.get("input_filename"),
        "created_at": result.get("created_at"),
    }

