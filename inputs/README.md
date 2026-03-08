# Input images

Place one or more images in this folder to run the pipeline. A lightweight sample image `example.png` is included for quick local runs.

**Example CLI run** (from repo root):

```bash
python -m src.cli.run \
  --input inputs/ \
  --out artifacts/runs \
  --run_id demo_run \
  --validate_schema \
  --framework anomalib \
  --model_name patchcore \
  --ckpt_path assets/models/model.ckpt
```

Outputs are written under `artifacts/runs/<run_id>/` (per-sample artifacts and `index.jsonl`).
