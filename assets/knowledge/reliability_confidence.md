# Reliability Confidence

Reliability confidence summarizes **how trustworthy the model’s decision is**,
not whether a defect truly exists.

It is designed as an **operational signal** to help decide  
*when automation is sufficient* and *when human review is required*.

Reliability confidence is especially important for handling **uncertain** cases.

---

## What reliability confidence represents

Reliability confidence combines multiple weak signals that indicate
how stable or interpretable the anomaly decision is:

- Distance between the anomaly score and the decision threshold
- Heatmap behavior (localized vs diffuse anomaly signal)
- Area ratio of highlighted regions

A **low confidence value does NOT mean a defect is present**.  
It means the model’s decision lies in a **low-trust region**.

---

## Heatmap energy concentration

Heatmap energy concentration describes **how localized the anomaly signal is**.

- **High concentration**
  - A small region dominates the anomaly signal
  - Suggests localized evidence (but still not defect confirmation)
- **Low concentration**
  - Anomaly signal is spread across the image
  - Often caused by illumination changes, reflections, texture repetition, or background patterns
  - Frequently associated with **uncertain** decisions

This signal helps distinguish:
- *“clear localized evidence”* vs
- *“diffuse, hard-to-interpret patterns”*

Note:
The concentration value is relative and implementation-dependent.
Only comparative interpretation within the same run or configuration is meaningful.

---

## Interpretation guidelines

The following ranges are indicative examples; exact boundaries may vary by configuration.

| Confidence range | Operational meaning |
|-----------------|--------------------|
| **High (> 0.8)** | Decision is stable and consistent |
| **Medium (0.5 – 0.8)** | Borderline or ambiguous; review recommended |
| **Low (< 0.5)** | Low-trust decision region; human review required |

Low confidence decisions are intentionally paired with the **uncertain** label.

---

## Relationship to triage

Reliability confidence is **one input** to the triage layer.

Typical patterns:
- Score near threshold + low confidence → **uncertain**
- Diffuse heatmap + low concentration → **uncertain**
- Stable score + localized heatmap → more reliable decision

This design favors **conservative operation**:
it reduces false automation at the cost of higher uncertain rates.

---

## Important notes

- Reliability confidence is a **heuristic signal**, not a calibrated probability.
- It should never be interpreted as “defect likelihood”.
- It exists to support **human-in-the-loop decision making**, not to replace it.

Low confidence ≠ defect  
Low confidence = *review required*

Changes in the concentration metric implementation do not imply changes
in operational policy or triage behavior.

---

## Implementation note (for developers)

The exact implementation of heatmap concentration may evolve
(e.g., energy-based ratios or entropy-based measures).

The operational meaning described above remains unchanged
as long as the metric captures **localized vs diffuse anomaly behavior**.