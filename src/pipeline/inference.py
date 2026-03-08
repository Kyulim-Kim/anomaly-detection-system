from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Registry for anomalib models (anomalib 2.x API).
#
# The following models have been smoke-tested with the current output extraction logic:
#   - patchcore
#   - padim
#   - fastflow
#
# Other anomalib models in the registry may also work but should be verified,
# because output ordering may vary.
ANOMALIB_MODEL_REGISTRY = {
    "patchcore": ("anomalib.models.image.patchcore", "Patchcore"),
    "padim": ("anomalib.models.image.padim", "Padim"),
    "fastflow": ("anomalib.models.image.fastflow", "Fastflow"),
    "stfpm": ("anomalib.models.image.stfpm", "Stfpm"),
    "reverse_distillation": ("anomalib.models.image.reverse_distillation", "ReverseDistillation"),
    "dfm": ("anomalib.models.image.dfm", "Dfm"),
}


def _load_anomalib_model(model_name: str, ckpt_path: str):
    """Load an anomalib model by name from checkpoint. Raises ValueError if unknown, RuntimeError if load fails."""
    key = model_name.strip().lower()
    if key not in ANOMALIB_MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported anomalib model_name: {model_name!r}. "
            f"Supported: {list(ANOMALIB_MODEL_REGISTRY.keys())}"
        )
    module_path, class_name = ANOMALIB_MODEL_REGISTRY[key]
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls.load_from_checkpoint(ckpt_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load anomalib model {model_name!r} from {ckpt_path!r}: {e}") from e


@dataclass(frozen=True)
class InferenceOutput:
    """
    Model-agnostic inference output.
    score: decision score = quantile(anomaly_map_raw, 0.995), raw scale. Used for label.
    anomaly_map: [0,1] normalized map for overlay/heatmap only.
    raw_score: model image-level score (logging/debug only).
    """

    score: float
    label: str  # "normal" | "anomaly" | "uncertain"(optional)
    threshold: float
    anomaly_map: np.ndarray  # float32 (H,W) in [0,1]
    model_meta: Dict[str, str]
    raw_score: Optional[float] = None


# -----------------------------
# Utilities
# -----------------------------
def _normalize01(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(m.min())
    hi = float(m.max())
    if hi <= lo + 1e-8:
        return np.zeros_like(m, dtype=np.float32)
    m = (m - lo) / (hi - lo)
    return np.clip(m, 0.0, 1.0).astype(np.float32)

def _safe_float(x):
    """
    Convert model output (tensor/ndarray/scalar) to float. No 0-1 clamping.
    Returns None if conversion fails. For logging/debug only.
    """
    if x is None:
        return None
    try:
        v = float(np.asarray(x).reshape(-1)[0])
        return v
    except Exception:
        return None

def _ensure_hw(m: np.ndarray) -> np.ndarray:
    """
    Accepts possible anomaly_map shapes from various libs:
    - (H,W)
    - (1,H,W)
    - (1,1,H,W)
    Returns (H,W).
    """
    if m.ndim == 2:
        return m
    if m.ndim == 3:
        return m[0]
    if m.ndim == 4:
        return m[0, 0]
    raise ValueError(f"Unexpected anomaly_map shape: {m.shape}")


# -----------------------------
# Anomalib (Python API) runner
# -----------------------------
class _AnomalibRunner:
    def __init__(
        self,
        ckpt_path: Path,
        *,
        model_name: str,
        threshold: float,
        device: Optional[str] = None,
        input_size_hw: Optional[Tuple[int, int]] = None,  # (H,W) optional resize
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        self.model_name = model_name.strip().lower()
        self.framework = "anomalib"
        self.threshold = float(threshold)
        self.input_size_hw = input_size_hw

        # Lazy imports: keep base viewer env light
        import torch

        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load model via registry
        self.model = _load_anomalib_model(self.model_name, str(self.ckpt_path))
        self.model.to(self.device)
        self.model.eval()

        import anomalib as _anomalib

        self.model_meta = {
            "name": self.model_name,
            "version": getattr(_anomalib, "__version__", "unknown"),
            "framework": self.framework,
        }

    def _prep_tensor(self, img_rgb_uint8: np.ndarray) -> Any:
        """
        Generic preprocessing used for inference.

        Images are converted to float [0,1] tensors and optionally resized.
        This works for most anomalib models, but exact preprocessing used
        during training may differ (normalization, resizing, padding).

        If strict reproducibility is required, training transforms should
        be restored from the training configuration or checkpoint metadata.
        """
        # img_rgb_uint8: HWC uint8
        x = img_rgb_uint8
        if self.input_size_hw is not None:
            # PIL resize to avoid cv2 dependency
            from PIL import Image

            h, w = self.input_size_hw
            x = np.asarray(Image.fromarray(x).resize((w, h), resample=Image.BILINEAR))

        # to float [0,1], CHW
        x = x.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # CHW
        t = self.torch.from_numpy(x).unsqueeze(0)  # 1CHW
        return t.to(self.device)

  
    @staticmethod
    def _extract_from_output(out) -> Tuple[np.ndarray, Optional[float]]:
        """
        Extraction strategy:
        0) Common anomalib InferenceBatch pattern fast-path
        1) Generic robust extraction fallback

        Many anomalib models return an InferenceBatch-like sequence:
            out[0] -> image-level score
            out[1] -> predicted label
            out[2] -> anomaly_map
            out[3] -> predicted mask

        However this is not guaranteed across all models,
        so a generic recursive fallback extractor is provided.
        """
        import numpy as _np

        # ------------------------------------------------------------
        # 0) Common anomalib InferenceBatch pattern fast-path
        # ------------------------------------------------------------
        # We explicitly take score from out[0] and map from out[2]
        # to avoid accidentally picking out[1] (often 1.0) as the score.
        try:
            if hasattr(out, "__len__") and len(out) >= 3:
                score_raw = out[0]
                map_raw = out[2]

                # to numpy
                if hasattr(map_raw, "detach"):
                    m = map_raw.detach().cpu().float().numpy()
                else:
                    m = _np.asarray(map_raw, dtype=_np.float32)

                # Defensive: element 2 must be plausibly an anomaly map (ndim >= 2)
                if m.ndim < 2:
                    raise ValueError("element 2 not array-like with ndim >= 2")

                # normalize to (H,W)
                if m.ndim == 4 and m.shape[0] == 1 and m.shape[1] == 1:
                    chosen_map = m[0, 0].astype(_np.float32)
                elif m.ndim == 3 and m.shape[0] == 1:
                    chosen_map = m[0].astype(_np.float32)
                elif m.ndim == 2:
                    chosen_map = m.astype(_np.float32)
                else:
                    chosen_map = None

                if chosen_map is not None:
                    # score to float
                    if hasattr(score_raw, "detach"):
                        s_arr = score_raw.detach().cpu().float().numpy()
                    else:
                        s_arr = _np.asarray(score_raw)
                    score = float(_np.asarray(s_arr).reshape(-1)[0])
                    return chosen_map, score
        except Exception:
            # If anything unexpected happens, fall back to generic logic below.
            pass

        # ------------------------------------------------------------
        # 1) Generic robust extraction fallback
        # ------------------------------------------------------------
        #
        # Fallback extraction walks the output object recursively and
        # searches for:
        #   - scalar-like values (candidate scores)
        #   - array-like values convertible to (H,W) anomaly maps
        #
        # This ensures robustness if anomalib changes internal output
        # structures or if new models expose different fields.

        def to_numpy(x):
            if hasattr(x, "detach"):
                return x.detach().cpu().float().numpy()
            return _np.asarray(x)

        def is_scalar_like(a: _np.ndarray) -> bool:
            return a.ndim == 0 or (a.ndim == 1 and a.size == 1)

        def try_to_hw(a: _np.ndarray) -> Optional[_np.ndarray]:
            """
            Return (H,W) if possible else None.
            Accept:
            (H,W), (1,H,W), (1,1,H,W)
            Reject:
            (1,), scalars, weird shapes
            """
            if a.ndim == 2:
                return a
            if a.ndim == 3 and a.shape[0] == 1:
                return a[0]
            if a.ndim == 4 and a.shape[0] == 1 and a.shape[1] == 1:
                return a[0, 0]
            return None

        # Collect candidates
        map_candidates = []
        score_candidates = []

        def add_candidate(x):
            try:
                arr = to_numpy(x)
            except Exception:
                return
            if is_scalar_like(arr):
                score_candidates.append(arr)
            else:
                map_candidates.append(arr)

        def walk(obj):
            # dict
            if isinstance(obj, dict):
                for v in obj.values():
                    walk(v)
                return
            # list/tuple
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    walk(v)
                return
            # object with attributes: take a few common ones first
            for attr in [
                "anomaly_map", "anomaly_maps", "pred_mask", "pred_masks", "pred_map", "pred_maps",
                "pred_score", "pred_scores", "anomaly_score", "anomaly_scores", "score", "scores"
            ]:
                if hasattr(obj, attr):
                    walk(getattr(obj, attr))
            # finally treat obj itself as candidate
            add_candidate(obj)

        walk(out)

        # Pick anomaly map: first that can become (H,W)
        chosen_map = None
        for cand in map_candidates:
            hw = try_to_hw(cand)
            if hw is not None:
                chosen_map = hw.astype(_np.float32)
                break

        if chosen_map is None:
            # Helpful debug
            shapes = [getattr(c, "shape", None) for c in map_candidates[:10]]
            raise RuntimeError(f"Could not find a valid (H,W) anomaly map. Candidate shapes: {shapes}")

        # Pick score: first scalar-like
        score = None
        if score_candidates:
            try:
                score = float(_np.asarray(score_candidates[0]).reshape(-1)[0])
            except Exception:
                score = None

        return chosen_map, score

    def predict(self, img_rgb_uint8: np.ndarray) -> InferenceOutput:
        torch = self.torch

        with torch.inference_mode():
            x = self._prep_tensor(img_rgb_uint8)
            out = self.model(x)
            am_raw, score_raw = self._extract_from_output(out)

        am_raw_np = np.asarray(am_raw, dtype=np.float32)
        if am_raw_np.ndim != 2:
            am_raw_np = _ensure_hw(am_raw_np)

        decision_score = float(np.quantile(am_raw_np, 0.995))
        am01 = _normalize01(am_raw_np)
        raw_score_debug = _safe_float(score_raw)
        label = "anomaly" if decision_score >= self.threshold else "normal"

        return InferenceOutput(
            score=float(decision_score),
            label=label,
            threshold=float(self.threshold),
            anomaly_map=am01,
            model_meta=self.model_meta,
            raw_score=raw_score_debug,
        )


# Cache singleton runner (avoid re-loading ckpt per request)
_RUNNER: Optional[_AnomalibRunner] = None
_RUNNER_KEY: Optional[Tuple[str, str, str, float, Optional[str], Optional[Tuple[int, int]]]] = None


def run_inference(
    img_rgb_uint8: np.ndarray,
    *,
    framework: str,
    model_name: str,
    ckpt_path: str,
    threshold: float = 0.5,
    device: Optional[str] = None,
    input_size_hw: Optional[Tuple[int, int]] = None,
) -> InferenceOutput:
    """
    Run inference with the given framework and model. Supported: framework="anomalib".

    Parameters
    ----------
    img_rgb_uint8 : np.ndarray
        Input RGB image as HWC uint8.
    framework : str
        Backend (e.g. "anomalib").
    model_name : str
        Model name within the framework (e.g. patchcore, padim, fastflow).
    ckpt_path : str
        Path to checkpoint.
    threshold : float
        Decision threshold for label.
    device : Optional[str]
        "cpu" or "cuda". Default: auto.
    input_size_hw : Optional[Tuple[int,int]]
        Resize (H,W) before inference (optional). Leave None to keep original size.

    Returns
    -------
    InferenceOutput
    """
    global _RUNNER, _RUNNER_KEY

    if framework != "anomalib":
        raise ValueError(f"Unsupported framework: {framework!r}. Supported: anomalib.")

    key = (framework, model_name.strip().lower(), str(Path(ckpt_path)), float(threshold), device, input_size_hw)
    if _RUNNER is None or _RUNNER_KEY != key:
        _RUNNER = _AnomalibRunner(
            Path(ckpt_path),
            model_name=model_name,
            threshold=threshold,
            device=device,
            input_size_hw=input_size_hw,
        )
        _RUNNER_KEY = key

    return _RUNNER.predict(img_rgb_uint8)