from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from src.pipeline.inference import run_inference
from src.pipeline.evaluation import compute_evaluation
from src.pipeline.rag_llm import run_llm_uncertain, run_rag
from src.pipeline.calibration import ecdf_risk_prob
from src.pipeline.reliability import compute_reliability
from src.pipeline.triage import decide_final_label
from src.pipeline.xai_postprocess import compute_xai
from src.utils.time import utc_now_iso


@dataclass
class PipelineOutput:
    anomaly_map: np.ndarray
    result: Dict[str, Any]


def build_result_json(
    *,
    run_id: str,
    sample_id: str,
    input_filename: str,
    input_sha1: str,
    image_size_wh: Tuple[int, int],  # (w, h)
    prediction_label: str,
    prediction_score: float,
    prediction_threshold: float,
    model_meta: Dict[str, str],
    explainability: Dict[str, Any],
    reliability: Dict[str, Any],
    rag: Dict[str, Any],
    llm: Dict[str, Any],
    evaluation: Dict[str, Any],
    meta_debug: Optional[Dict[str, Any]] = None,
    triage: Optional[Dict[str, Any]] = None,
    prediction_probability: Optional[float] = None,
    prediction_risk_est: Optional[float] = None,
    gt_mask_path_rel: Optional[str] = None,
    debug_dir_name: Optional[str] = None,
    defect_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Construct `result.json` matching `schemas/result.schema.json`.

    Constraint:
    - `paths` are relative to sample directory (where result.json lives).
    """
    w, h = image_size_wh
    paths = {
        "original": "original.png",
        "heatmap": "heatmap.png",
        "overlay": "overlay.png",
        "result": "result.json",
        "gt_mask": gt_mask_path_rel,
        "anomaly_map_npy": "anomaly_map.npy",
        "debug_dir": debug_dir_name,
    }
    meta = {
        "input_filename": input_filename,
        "input_sha1": input_sha1,
        "image_size_wh": [int(w), int(h)],
        "tags": [],
        **({"debug": meta_debug} if meta_debug is not None else {}),
    }
    if defect_type is not None:
        meta["defect_type"] = defect_type
    return {
        "schema_version": "1.0",  # keep simple; bump when breaking changes
        "sample_id": sample_id,
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "paths": paths,
        "prediction": {
            "label": prediction_label,
            "score": float(prediction_score),
            "threshold": float(prediction_threshold),
            "probability": None,
            "score_method": "anomaly_map_quantile",
            "score_quantile": 0.995,
            "task": "image",
            "model": model_meta,
            **({"risk_est": float(prediction_risk_est)} if prediction_risk_est is not None else {}),
        },
        "explainability": explainability,
        "reliability": reliability,
        "rag": rag,
        "llm": llm,
        "evaluation": evaluation,
        **({"triage": triage} if triage is not None else {}),
        "meta": meta,
    }


def run_pipeline(
    original_pil_rgb: Image.Image,
    *,
    seed: int,
    run_id: str,
    sample_id: str,
    input_filename: str,
    input_sha1: str,
    gt_label_display: Optional[str] = None,
    has_gt: bool = False,
    defect_type: Optional[str] = None,
    gt_mask_path_rel: Optional[str] = None,
    debug_dir_name: Optional[str] = None,
    threshold: float = 0.5,
    framework: str,
    model_name: str,
    ckpt_path: str,
    device: Optional[str],
    triage_margin_eps: float = 0.03,
    triage_conf_eps: float = 0.55,
    triage_area_hi: float = 0.12,
    triage_concentration_lo: float = 0.25,
    risk_ecdf: Optional[Dict[str, Any]] = None,
    concentration_calib: Optional[Dict[str, Any]] = None,
    rag_docs_dir: str = "assets/knowledge",
    rag_top_k: int = 5,
    rag_force_rebuild: bool = False,
) -> PipelineOutput:
    """
    End-to-end pipeline producing:
    - anomaly_map (np)
    - heatmap stats + hotspots (dict)
    - result.json dict (schema-compliant)

    NOTE: model replacement point is `src/pipeline/inference.py`.
    """
    w, h = original_pil_rgb.size

    # 1) inference
    img = np.asarray(original_pil_rgb.convert("RGB"), dtype=np.uint8)
    inf = run_inference(
        img,
        framework=framework,
        model_name=model_name,
        ckpt_path=ckpt_path,
        threshold=threshold,
        device=device,
    )

    # 2) explainability (xai postprocess)
    xai_out = compute_xai(inf.anomaly_map)
    heatmap_stats_xai = asdict(xai_out.heatmap_stats)

    explainability = {
        "heatmap_stats": heatmap_stats_xai,
        "hotspots": [{"bbox_xyxy": list(hs.bbox_xyxy), "score": float(hs.score)} for hs in xai_out.hotspots],
        "notes": xai_out.notes,
    }

    # 3) reliability (based on score/threshold + xai stats)
    heatmap_stats_rel = {
        "max_intensity": float(heatmap_stats_xai["max"]),
        "mean_intensity": float(heatmap_stats_xai["mean"]),
        "area_ratio": float(getattr(xai_out, "area_ratio", 0.0)),
    }

    rel_out = compute_reliability(
        score=float(inf.score),
        threshold=float(inf.threshold),
        heatmap_stats=heatmap_stats_rel,
        heatmap=inf.anomaly_map,
        concentration_calib=concentration_calib,
    )

    reliability = {
        "confidence": float(rel_out.confidence),
        "signals": rel_out.signals,
        "notes": rel_out.notes,
    }

    base_label = inf.label
    final_label, triage_reasons = decide_final_label(
        score=float(inf.score),
        threshold=float(inf.threshold),
        reliability_confidence=float(rel_out.confidence),
        reliability_signals=rel_out.signals,
        xai_hotspots=explainability.get("hotspots", []),
        margin_eps=triage_margin_eps,
        conf_eps=triage_conf_eps,
        area_hi=triage_area_hi,
        concentration_lo=triage_concentration_lo,
    )
    triage_obj = {
        "base_label": base_label,
        "final_label": final_label,
        "reasons": triage_reasons,
    }


    # 4) RAG + LLM: only for uncertain (cost saving + human-in-the-loop)
    if final_label == "uncertain":
        query_parts = [
            f"defect_type: {defect_type}" if defect_type else "defect_type: (unknown)",
            f"final_label: {final_label}",
            f"score: {inf.score:.4f}",
            f"threshold: {inf.threshold:.4f}",
            f"heatmap p95: {heatmap_stats_xai.get('p95', 0):.4f}",
            f"heatmap max: {heatmap_stats_xai.get('max', 0):.4f}",
            f"area_ratio: {getattr(xai_out, 'area_ratio', 0):.4f}",
        ]
        hotspots_list = explainability.get("hotspots", [])[:5]
        query_parts.append("hotspots: " + "; ".join(f"bbox{list(h.get('bbox_xyxy', []))} score={h.get('score', 0):.3f}" for h in hotspots_list))
        query = " | ".join(query_parts)
        rag_out = run_rag(query, docs_dir=rag_docs_dir, top_k=rag_top_k, force_rebuild=rag_force_rebuild)
        concentration = float((rel_out.signals or {}).get("heatmap_concentration", 0))
        llm_out = run_llm_uncertain(
            score=float(inf.score),
            threshold=float(inf.threshold),
            confidence=float(rel_out.confidence),
            triage_reasons=triage_reasons,
            heatmap_stats=heatmap_stats_xai,
            area_ratio=float(getattr(xai_out, "area_ratio", 0)),
            concentration=concentration,
            hotspots=hotspots_list,
            defect_type=defect_type,
            contexts=rag_out.contexts_for_llm,
        )
        rag = {"context_used": rag_out.context_used, "contexts": rag_out.contexts, "notes": rag_out.notes}
        llm = {"summary": llm_out.summary, "explanation": llm_out.explanation, "notes": llm_out.notes}
    else:
        rag = {"context_used": False, "contexts": [], "notes": ""}
        llm = {"summary": "", "explanation": "", "notes": "skipped_non_uncertain"}

    # 5) evaluation (uses final_label so uncertain -> error_type null)
    evaluation = compute_evaluation(
        pred_label=final_label,
        gt_label_display=gt_label_display,
        has_gt=has_gt,
    )

    # 6) Risk (est.) = ECDF percentile; stored as prediction.risk_est (internal/debug). prediction.probability is deprecated and always null.
    risk = ecdf_risk_prob(risk_ecdf, float(inf.score)) if risk_ecdf else None

    # 7) build result.json (prediction.label = final_label; triage; risk_est when available; probability always null)
    meta_debug = {"raw_score_debug": inf.raw_score, "decision_score_q": 0.995}
    result = build_result_json(
        run_id=run_id,
        sample_id=sample_id,
        input_filename=input_filename,
        input_sha1=input_sha1,
        image_size_wh=(w, h),
        prediction_label=final_label,
        prediction_score=inf.score,
        prediction_threshold=inf.threshold,
        model_meta=inf.model_meta,
        explainability=explainability,
        reliability=reliability,
        rag=rag,
        llm=llm,
        evaluation=evaluation,
        meta_debug=meta_debug,
        triage=triage_obj,
        prediction_risk_est=risk,
        gt_mask_path_rel=gt_mask_path_rel,
        debug_dir_name=debug_dir_name,
        defect_type=defect_type,
    )

    return PipelineOutput(
        anomaly_map=inf.anomaly_map,
        result=result,
    )