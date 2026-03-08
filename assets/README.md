# Assets

This directory stores external resources used by the pipeline.

Typical contents include:

- model configuration files
- trained model checkpoints
- optional knowledge files used for RAG or documentation

Example structure:

```
assets/
├─ configs/
│  └─ model.yaml
├─ models/
│  └─ model.ckpt
└─ knowledge/
   └─ rag.md
```

## configs/
Configuration files for model inference or experiment settings.

Examples:
- `model.yaml`
- `model.json`

## models/
Trained model checkpoints used for inference.

Examples:
- `patchcore.ckpt`
- `padim.ckpt`

Note:
Model files are usually **not committed to Git** due to size.

## knowledge/
Optional reference documents used for retrieval or explanation.

Examples:
- `rag.md`
- domain notes
- inspection guidelines