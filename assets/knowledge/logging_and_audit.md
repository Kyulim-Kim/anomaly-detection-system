# Logging and Audit Policy

All inference results are logged for **operational review and auditing purposes**.

Each log entry includes:
- Input metadata (file identifiers, timestamps)
- Decision score and decision threshold
- Final label (normal / anomaly / uncertain)
- Heatmap and region-level evidence
- Explanatory context (when applicable)

## Purpose of logging

Logs are used exclusively for:
- Post-incident analysis and root cause review
- Threshold and triage parameter tuning
- Model behavior monitoring and validation
- Human-in-the-loop audit and decision traceability

Logged data is **not** used to retroactively change model decisions
or to automate decisions outside the inference pipeline.

## Audit and responsibility

This system supports **human-in-the-loop auditing**:
- Automated decisions are made at inference time
- Uncertain cases are explicitly flagged for human review
- Logged records provide context for review, not automatic correction

Final operational decisions remain the responsibility of human operators
when the system reports low-confidence or uncertain outcomes.

## Data and privacy

This system does not store personal or user-identifiable data.
All logged information is limited to:
- Image-derived features
- Model outputs
- Operational metadata required for analysis

No personal identifiers are collected, inferred, or retained.