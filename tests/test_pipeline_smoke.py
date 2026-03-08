"""Lightweight smoke test for the pipeline (no real checkpoint)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.pipeline.inference import InferenceOutput
from src.pipeline.pipeline import run_pipeline
from src.utils.image_ops import load_image_rgb

# Path to optional sample image (from repo root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_IMAGE_PATH = _PROJECT_ROOT / "inputs" / "example.png"


def _make_fake_inference_output(*, h: int = 32, w: int = 32) -> InferenceOutput:
    """Deterministic fake inference: small anomaly map with one bright patch."""
    anomaly_map = np.zeros((h, w), dtype=np.float32)
    anomaly_map[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0.9  # bright patch
    return InferenceOutput(
        score=0.4,
        label="normal",
        threshold=0.5,
        anomaly_map=anomaly_map,
        model_meta={"name": "test", "version": "0", "framework": "test"},
        raw_score=None,
    )


@pytest.fixture
def monkeypatch_run_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace run_inference in the pipeline with a fake that returns a deterministic output."""
    fake_out = _make_fake_inference_output()

    def fake_run_inference(*args: object, **kwargs: object) -> InferenceOutput:
        return fake_out

    monkeypatch.setattr("src.pipeline.pipeline.run_inference", fake_run_inference)


def test_pipeline_smoke_skips_without_example_image() -> None:
    """If inputs/example.png is missing, skip with a clear message."""
    if not EXAMPLE_IMAGE_PATH.is_file():
        pytest.skip(
            "inputs/example.png not found. "
            "Add a sample image to inputs/ to run the pipeline smoke test."
        )


@pytest.mark.usefixtures("monkeypatch_run_inference")
def test_pipeline_smoke_returns_without_crashing() -> None:
    """Run pipeline on inputs/example.png (with mocked inference); no crash."""
    if not EXAMPLE_IMAGE_PATH.is_file():
        pytest.skip(
            "inputs/example.png not found. "
            "Add a sample image to inputs/ to run the pipeline smoke test."
        )
    original = load_image_rgb(EXAMPLE_IMAGE_PATH)
    out = run_pipeline(
        original,
        seed=12345,
        run_id="smoke_run",
        sample_id="smoke_sample",
        input_filename="example.png",
        input_sha1="abc000",
        threshold=0.5,
    )
    assert out is not None
    assert hasattr(out, "anomaly_map")
    assert hasattr(out, "result")
    assert out.result is not None
    assert out.anomaly_map.ndim == 2
    assert out.anomaly_map.shape[0] > 0 and out.anomaly_map.shape[1] > 0
    assert out.result.get("prediction", {}).get("label") in ("normal", "anomaly", "uncertain")
    assert "sample_id" in out.result
    assert "run_id" in out.result
    assert "explainability" in out.result
    assert "reliability" in out.result
    assert "triage" in out.result
