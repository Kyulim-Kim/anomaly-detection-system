# Operator Action Guidelines

This document defines **operator actions based on system output labels**.
The goal is to ensure safe operation while preserving human responsibility
in low-confidence situations.

---

## When label = anomaly

- Inspect highlighted regions and associated heatmaps
- Compare findings with known defect characteristics
- Escalate to QA or process owners if a defect is visually confirmed

Note:
An *anomaly* label indicates strong anomaly evidence,
but it does not replace human confirmation of a true defect.

---

## When label = normal

- No immediate action required
- Samples may still be selected for periodic audit or spot checks
- Normal classification does not imply absolute certainty,
only sufficient confidence for automated acceptance

---

## When label = uncertain

- **Mandatory manual inspection**
- Do NOT auto-accept or auto-reject the sample
- Review image quality, heatmap patterns, and highlighted regions
- Record the inspection outcome for operational analysis and system tuning

The *uncertain* label represents an **intentional low-trust decision region**.
It is designed to prevent over-automation when evidence is ambiguous.

---

## Operational principle

This system is designed to:
- Reduce repetitive inspection workload
- Prioritize human review where model confidence is limited
- Support conservative decision-making in safety- or quality-critical workflows

The system assists operators —  
it does **not** eliminate human judgment or accountability.