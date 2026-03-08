from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from PIL import Image

from src.storage.artifacts import (
    append_jsonl,
    index_record_for_sample,
    init_run_dir,
    load_run_meta,
    write_run_meta,
    write_sample_artifacts,
)
from src.pipeline.calibration import build_ecdf
from src.pipeline.pipeline import run_pipeline
from src.pipeline.reliability import raw_heatmap_concentration
from src.pipeline.threshold import resolve_threshold
from src.pipeline.inference import run_inference
from src.utils.hashing import sha1_file, stable_sample_id
from src.utils.image_ops import alpha_blend, jet_colormap, load_image_rgb, np_float01_to_pil, pil_to_np_float01
from src.utils.time import utc_now_iso


def get_git_commit() -> Optional[str]:
    """Return current git commit hash if available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def iter_images(input_dir: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p

def resize_anomaly_map_to_image(anomaly_map01: np.ndarray, pil_img: Image.Image) -> np.ndarray:
    h, w = pil_img.size[1], pil_img.size[0]  # PIL: (W,H)
    m = (anomaly_map01 * 255.0).astype(np.uint8)
    m_resized = np.asarray(Image.fromarray(m).resize((w, h), resample=Image.BILINEAR)).astype(np.float32) / 255.0
    return np.clip(m_resized, 0.0, 1.0).astype(np.float32)

def render_heatmap_and_overlay(original_rgb: Image.Image, anomaly_map01: np.ndarray, alpha: float = 0.45) -> tuple[Image.Image, Image.Image]:
    heat_rgb01 = jet_colormap(anomaly_map01)
    heatmap = np_float01_to_pil(heat_rgb01)

    base01 = pil_to_np_float01(original_rgb)
    overlay01 = alpha_blend(base01, heat_rgb01, alpha=alpha)
    overlay = np_float01_to_pil(overlay01)
    return heatmap, overlay


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run anomaly pipeline and write standard artifacts.")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing images.")
    parser.add_argument("--out", type=str, required=True, help="Output root folder, e.g. artifacts/runs")
    parser.add_argument("--run_id", type=str, required=True, help="Run id, e.g. demo_run")
    parser.add_argument("--validate_schema", action="store_true", help="Validate result.json before saving.")
    parser.add_argument("--schema_path", type=str, default="schemas/result.schema.json", help="Schema path for validation.")
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="Optional GT directory. For MVTec: looks for <gt_dir>/<defect_type>/<stem>_mask.png or <stem>.png.",
    )
    parser.add_argument(
        "--debug_dir_name",
        type=str,
        default=None,
        help="If set, create this subdir under each sample and set paths.debug_dir to it.",
    )
    parser.add_argument(
        "--threshold_mode",
        type=str,
        choices=["fixed", "normal_p995"],
        default="fixed",
        help="Threshold policy: fixed (use --threshold) or normal_p995 (use --threshold_normal_dir).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold value when --threshold_mode=fixed.",
    )
    parser.add_argument(
        "--threshold_normal_dir",
        type=str,
        default=None,
        help="Directory of normal images to compute 99.5%% quantile threshold when --threshold_mode=normal_p995.",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="anomalib",
        help="Model framework backend (default: anomalib).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="patchcore",
        help="Model name within the selected framework (default: patchcore).",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to model checkpoint (used for inference and for normal_p995 score collection).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference: cpu or cuda.",
    )
    parser.add_argument("--uncertain_margin_eps", type=float, default=0.03, help="Triage: margin below this → uncertain (borderline).")
    parser.add_argument("--uncertain_conf_eps", type=float, default=0.55, help="Triage: reliability confidence below this → uncertain.")
    parser.add_argument("--uncertain_area_hi", type=float, default=0.12, help="Triage: area_ratio above this → uncertain.")
    parser.add_argument("--uncertain_concentration_lo", type=float, default=0.25, help="Triage: heatmap concentration below this → uncertain.")
    parser.add_argument("--enable_risk_ecdf", action="store_true", default=True, help="When threshold_mode=normal_p995, build ECDF and set prediction.probability (Risk est.).")
    parser.add_argument("--no_enable_risk_ecdf", action="store_false", dest="enable_risk_ecdf", help="Disable Risk (est.) ECDF calibration.")
    parser.add_argument("--risk_ecdf_max_points", type=int, default=5000, help="Max points to keep in ECDF scores_sorted (downsample).")
    args = parser.parse_args(argv)

    input_dir = Path(args.input)
    out_root = Path(args.out)
    run_id = args.run_id
    gt_dir = Path(args.gt_dir) if args.gt_dir is not None else None
    threshold_mode = args.threshold_mode
    threshold_normal_dir = Path(args.threshold_normal_dir) if args.threshold_normal_dir else None
    framework = args.framework
    model_name = args.model_name
    ckpt_path = args.ckpt_path
    device = args.device

    # Dataset meta: parse common MVTec-style paths like datasets/mvtec/<category>/<split>/<subset>
    category = None
    split = None
    subset = None
    try:
        parts = list(input_dir.parts)
        if len(parts) >= 3 and "mvtec" in parts:
            category = parts[-3]
            split = parts[-2]
            subset = parts[-1]
    except Exception:
        pass
    dataset_meta = {
        "name": "mvtec_ad",
        "category": category,
        "split": split,
        "subset": subset,
        "input_dir": str(input_dir),
    }
    code_meta = {"git_commit": get_git_commit()}
    repro_meta = {"seed_policy": "sha1_image_hash"}
    config_meta = {
        "validate_schema": bool(args.validate_schema),
        "device": device,
        "threshold_mode": threshold_mode,
        "triage_margin_eps": float(args.uncertain_margin_eps),
        "triage_conf_eps": float(args.uncertain_conf_eps),
        "triage_area_hi": float(args.uncertain_area_hi),
        "triage_concentration_lo": float(args.uncertain_concentration_lo),
        "enable_risk_ecdf": bool(args.enable_risk_ecdf),
        "risk_ecdf_max_points": int(args.risk_ecdf_max_points),
    }

    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    if threshold_mode == "normal_p995" and (threshold_normal_dir is None or not threshold_normal_dir.exists()):
        raise SystemExit("--threshold_mode=normal_p995 requires --threshold_normal_dir to an existing directory.")

    if ckpt_path is None:
        raise SystemExit(
            "--ckpt_path is required. Provide a trained model checkpoint."
        )

    # Optional checkpoint hash for run-level provenance
    ckpt_sha1 = None
    try:
        ckpt_p = Path(ckpt_path)
        if ckpt_p.exists() and ckpt_p.is_file():
            ckpt_sha1 = sha1_file(ckpt_p)
    except Exception:
        ckpt_sha1 = None

    # Model config for run_meta (and inference)
    model_config = {
        "framework": framework,
        "name": model_name,
        "ckpt_path": ckpt_path,
        "ckpt_sha1": ckpt_sha1,
        "device": device,
        "input_size_hw": None,
    }

    run_dir = init_run_dir(out_root, run_id)
    schema_path = Path(args.schema_path) if args.validate_schema else None
    if schema_path is not None and not schema_path.exists():
        raise SystemExit(f"Schema not found: {schema_path}")

    existing_meta = load_run_meta(run_dir)
    enable_risk_ecdf = args.enable_risk_ecdf
    risk_ecdf_max_points = args.risk_ecdf_max_points

    # Reuse run_meta (threshold + calibration) when available for normal_p995
    if threshold_mode == "normal_p995" and existing_meta and existing_meta.get("calibration", {}).get("ecdf") and enable_risk_ecdf:
        resolved_threshold = existing_meta["threshold_policy"]["threshold"]
        risk_ecdf = existing_meta["calibration"]["ecdf"]
        run_meta = existing_meta
    else:
        risk_ecdf = None
        if threshold_mode == "normal_p995":
            normal_scores: List[float] = []
            raw_concentrations: List[float] = []
            for img_path in iter_images(threshold_normal_dir):
                original = load_image_rgb(img_path)
                img = np.asarray(original.convert("RGB"), dtype=np.uint8)
                inf = run_inference(
                    img,
                    framework=framework,
                    model_name=model_name,
                    ckpt_path=ckpt_path,
                    threshold=0.5,
                    device=device,
                )
                normal_scores.append(inf.score)
                raw_concentrations.append(raw_heatmap_concentration(inf.anomaly_map))
            resolved_threshold = resolve_threshold(
                mode="normal_p995",
                normal_scores=normal_scores,
                quantile=0.995,
            )
            threshold_policy = {
                "mode": "normal_p995",
                "threshold": resolved_threshold,
                "threshold_value": resolved_threshold,
                "decision_score_q": 0.995,
                "normal_quantile_p": 0.995,
                "threshold_normal_dir": str(threshold_normal_dir),
                "quantile": 0.995,
                "n_normal_samples": len(normal_scores),
            }
            # Same normal reference set for reliability: heatmap_concentration quantile min-max
            if raw_concentrations:
                c_lo = float(np.percentile(raw_concentrations, 5))
                c_hi = float(np.percentile(raw_concentrations, 95))
                reliability_policy = {
                    "heatmap_concentration_calibration": {
                        "method": "quantile_minmax",
                        "q_lo": 0.05,
                        "q_hi": 0.95,
                        "c_lo": c_lo,
                        "c_hi": c_hi,
                        "reference_dir": str(threshold_normal_dir),
                        "num_samples": len(raw_concentrations),
                    }
                }
            else:
                reliability_policy = {}
            if enable_risk_ecdf and normal_scores:
                ecdf = build_ecdf(normal_scores, max_points=risk_ecdf_max_points)
                risk_ecdf = ecdf
                calibration = {"method": "ecdf_normal", "ecdf": ecdf}
            else:
                calibration = {}
            run_meta = {
                "run_id": run_id,
                "created_at": utc_now_iso(),
                "dataset": dataset_meta,
                "pipeline": {"name": "industrial_anomaly_pipeline", "version": "v1"},
                "model": model_config,
                "code": code_meta,
                "reproducibility": repro_meta,
                "config": config_meta,
                "threshold_policy": threshold_policy,
                "calibration": calibration,
                **({"reliability_policy": reliability_policy} if reliability_policy else {}),
            }
        else:
            resolved_threshold = resolve_threshold(mode="fixed", fixed_value=args.threshold)
            threshold_policy = {
                "mode": "fixed",
                "threshold": resolved_threshold,
                "threshold_value": resolved_threshold,
                "decision_score_q": 0.995,
            }
            run_meta = {
                "run_id": run_id,
                "created_at": utc_now_iso(),
                "dataset": dataset_meta,
                "pipeline": {"name": "industrial_anomaly_pipeline", "version": "v1"},
                "model": model_config,
                "code": code_meta,
                "reproducibility": repro_meta,
                "config": config_meta,
                "threshold_policy": threshold_policy,
            }
        write_run_meta(run_dir, run_meta)

    index_path = run_dir / "index.jsonl"

    def infer_defect_type(img_path: Path) -> str:
        """MVTec-style: .../test/broken_small/000.png -> 'broken_small'; .../test/good/000.png -> 'good'."""
        return img_path.parent.name

    def find_gt_mask(gt_dir_path: Path, defect_type: str, stem: str) -> Optional[Path]:
        """Look for {gt_dir}/{defect_type}/{stem}_mask.png then {gt_dir}/{defect_type}/{stem}.png."""
        sub = gt_dir_path / defect_type
        if not sub.exists():
            return None
        for name in (f"{stem}_mask.png", f"{stem}.png"):
            p = sub / name
            if p.is_file():
                return p
        return None

    n = 0
    debug_dir_name = args.debug_dir_name

    for img_path in iter_images(input_dir):
        sha1 = sha1_file(img_path)
        sample_id = stable_sample_id(img_path, sha1)

        original = load_image_rgb(img_path)
        w, h = original.size
        seed = int(sha1[:8], 16)  # deterministic per image

        defect_type = infer_defect_type(img_path)
        gt_label_display = "good" if defect_type == "good" else defect_type

        gt_mask_pil: Optional[Image.Image] = None
        gt_mask_path_rel: Optional[str] = None
        has_gt = False
        if gt_dir is not None and gt_dir.exists():
            stem = img_path.stem
            gt_path = find_gt_mask(gt_dir, defect_type, stem)
            if gt_path is not None:
                try:
                    gt_mask_pil = load_image_rgb(gt_path)
                    has_gt = True
                    gt_mask_path_rel = "gt_mask.png"
                except Exception:
                    pass

        concentration_calib = (run_meta.get("reliability_policy") or {}).get("heatmap_concentration_calibration")
        pipe_out = run_pipeline(
            original,
            seed=seed,
            run_id=run_id,
            sample_id=sample_id,
            input_filename=img_path.name,
            input_sha1=sha1,
            gt_label_display=gt_label_display,
            has_gt=has_gt,
            defect_type=defect_type,
            gt_mask_path_rel=gt_mask_path_rel,
            debug_dir_name=debug_dir_name,
            threshold=resolved_threshold,
            framework=framework,
            model_name=model_name,
            ckpt_path=ckpt_path,
            device=device,
            triage_margin_eps=args.uncertain_margin_eps,
            triage_conf_eps=args.uncertain_conf_eps,
            triage_area_hi=args.uncertain_area_hi,
            triage_concentration_lo=args.uncertain_concentration_lo,
            risk_ecdf=risk_ecdf,
            concentration_calib=concentration_calib,
        )
        anomaly_map01 = pipe_out.anomaly_map
        result = pipe_out.result

        anomaly_map01 = resize_anomaly_map_to_image(anomaly_map01, original)
        heatmap, overlay = render_heatmap_and_overlay(original, anomaly_map01, alpha=0.45)

        # Write artifacts
        write_sample_artifacts(
            run_dir,
            sample_id,
            original=original,
            heatmap=heatmap,
            overlay=overlay,
            result=result,
            anomaly_map=anomaly_map01,
            gt_mask=gt_mask_pil,
            validate_schema_path=schema_path,
        )

        # Append index record
        rec = index_record_for_sample(run_id, sample_id, result)
        append_jsonl(index_path, rec)
        n += 1

    print(f"Wrote {n} samples to {run_dir}")
    print(f"Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

