# Uncertain Triage Policy

This document defines the **uncertain triage policy** for the anomaly detection system.

The purpose of this policy is to explicitly handle cases where the system’s
decision confidence is insufficient for safe automation.

Uncertain is **not an error** and **not a defect decision**.
It is an intentional outcome designed to enforce human review.

---

## Two-stage decision structure

The system applies a **two-stage decision process**:

### 1. Base decision (score-based)
- **anomaly** if anomaly score ≥ threshold
- **normal** otherwise

This decision is derived purely from the anomaly score
relative to the calibrated threshold.

### 2. Uncertain triage override (reliability-based)
The base decision may be overridden and replaced with **uncertain**
when the decision reliability is insufficient.

The triage layer does **not** change the anomaly score or threshold.
It only determines whether the base decision is trustworthy enough
to be acted upon automatically.

---

## When uncertain may be triggered

Uncertain may be triggered when **one or more conditions**
indicate insufficient decision reliability.

Typical signals include:

- **Borderline margin**  
  The anomaly score lies close to the decision threshold,
  making the classification sensitive to noise or minor variation.

- **Low reliability confidence**  
  The combined reliability signals indicate a low-trust decision region.

- **Unusually large anomaly area**  
  Highlighted regions cover a broad portion of the image,
  reducing spatial specificity of the evidence.

- **Diffuse heatmap patterns**  
  Anomaly evidence is spread across the image without a clear localized hotspot,
  often caused by illumination changes, texture repetition, or background effects.

These signals do **not** imply that a defect exists.
They indicate that the system cannot confidently support automation.

---

## Meaning of the uncertain label

An **uncertain** label means:

- The system cannot reliably decide between normal and anomaly
- **Manual inspection is required**
- **No automatic acceptance or rejection should occur**

Uncertain is intentionally conservative.
It favors safety and interpretability over full automation.

---

## Relationship to other system outputs

- Uncertain does **not** replace the anomaly score or threshold.
- It is evaluated **after** the base score-based decision.
- Reliability confidence and heatmap signals are used as **supporting evidence**,
  not as standalone decision criteria.

The final output label reflects this triage outcome.

---

## Operational intent

The uncertain mechanism exists to:

- Prevent overconfident automation in low-trust regions
- Surface ambiguous samples for human review
- Support stable deployment under limited calibration data

High uncertain rates do **not** necessarily indicate poor model performance.
They often reflect conservative operating conditions,
threshold choices, or challenging imaging environments.

---

## Implementation note (for maintainers)

Specific numeric cutoffs (e.g., confidence thresholds, area ratios)
are defined as **operational parameters**.

These values may be adjusted or calibrated per run or dataset
without changing the intent of the triage policy described here.