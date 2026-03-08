from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, label, find_objects, binary_dilation


@dataclass(frozen=True)
class HeatmapStats:
    min: float
    max: float
    mean: float
    p95: float


@dataclass(frozen=True)
class Hotspot:
    bbox_xyxy: Tuple[int, int, int, int]
    score: float  # keep as mean score inside region


@dataclass(frozen=True)
class XaiOutput:
    heatmap_stats: HeatmapStats
    hotspots: List[Hotspot]
    area_ratio: float
    notes: str = ""


def _robust_normalize(m: np.ndarray, *, p_lo: float = 50.0, p_hi: float = 99.5) -> np.ndarray:
    """Percentile clip then normalize to [0, 1]."""
    m = m.astype(np.float32)
    lo = np.percentile(m, p_lo)
    hi = np.percentile(m, p_hi)
    if hi <= lo + 1e-8:
        return np.zeros_like(m, dtype=np.float32)
    m = np.clip(m, lo, hi)
    m = (m - lo) / (hi - lo)
    return m


def compute_xai(
    anomaly_map01: np.ndarray,
    *,
    blur_sigma: float = 2.0,
    p_lo: float = 50.0,
    p_hi: float = 99.5,
    th_high: float = 0.85,
    th_low: float = 0.65,
    min_area: int = 80,
    top_k: int = 5,
) -> XaiOutput:
    """
    Better XAI postprocess:
    - robust normalization (percentile clip)
    - optional gaussian smoothing
    - dual-threshold region extraction (seed & grow)
    - connected components with area filtering
    """
    m_raw = np.clip(anomaly_map01.astype(np.float32), 0.0, 1.0)

    # robust normalize (prevents wide dull blobs dominating)
    m = _robust_normalize(m_raw, p_lo=p_lo, p_hi=p_hi)

    # smooth (reduces speckle; helps cleaner contours)
    if blur_sigma and blur_sigma > 0:
        m = gaussian_filter(m, sigma=float(blur_sigma))

    p95 = float(np.quantile(m, 0.95))
    stats = HeatmapStats(min=float(m.min()), max=float(m.max()), mean=float(m.mean()), p95=p95)

    # dual threshold: seeds then grow to low-th region
    seed = m >= float(th_high)
    grow = m >= float(th_low)

    # grow seeds within grow-mask (one simple way: dilate then AND)
    # (repeat dilation a few times implicitly via iterations)
    region = seed.copy()
    if np.any(seed):
        for _ in range(10):
            region = binary_dilation(region)
            region = region & grow

    # connected components
    hotspots: List[Hotspot] = []
    if np.any(region):
        cc, n = label(region)
        slices = find_objects(cc)

        comps = []
        for idx, slc in enumerate(slices, start=1):
            if slc is None:
                continue
            ys, xs = np.where(cc[slc] == idx)
            area = int(len(xs))
            if area < min_area:
                continue

            y0 = int(slc[0].start + ys.min())
            y1 = int(slc[0].start + ys.max())
            x0 = int(slc[1].start + xs.min())
            x1 = int(slc[1].start + xs.max())

            comp_mask = (cc == idx)
            score = float(m[comp_mask].mean())
            peak = float(m[comp_mask].max())
            comps.append((score, peak, area, (x0, y0, x1, y1)))

        # sort by score (or peak) and take top_k
        comps.sort(key=lambda t: (t[0], t[1]), reverse=True)
        for score, peak, area, bbox in comps[:top_k]:
            hotspots.append(Hotspot(bbox_xyxy=bbox, score=score))

    notes = (
        f"robust_norm(p_lo={p_lo},p_hi={p_hi}), blur_sigma={blur_sigma}, "
        f"dual_th(th_high={th_high},th_low={th_low}), min_area={min_area}, top_k={top_k}"
    )

    area_ratio = float(region.mean()) if region.size else 0.0
    return XaiOutput(heatmap_stats=stats, hotspots=hotspots, area_ratio=area_ratio,notes=notes)