# Uncertain Decision Examples

This document illustrates **typical operational scenarios**
where the system intentionally outputs the label **"uncertain"**.

These examples are not failure cases.
They demonstrate how the system avoids overconfident automation
in low-trust decision regions.

---

## Example 1: Normal-looking Image with Borderline Score

### Observed symptoms
- Visual inspection shows no obvious defect
- Decision score is slightly above the threshold
- Heatmap shows weak, diffuse anomaly evidence across a wide area

### System behavior
- Base label (score vs threshold): anomaly
- Reliability confidence: low
- Final label after triage: uncertain

### Interpretation
This pattern often occurs when:
- background appearance shifts slightly,
- illumination conditions vary,
- or normal structural patterns resemble rare features.

Although the score exceeds the threshold,
the lack of localized evidence reduces decision trust.

### Operational guidance
This case should be reviewed manually.
It does **not** imply a confirmed defect.

---

## Example 2: Visible Defect with Diffuse Heatmap

### Observed symptoms
- An anomaly is visible to the human eye
- Heatmap highlights a broad region rather than a compact hotspot
- Highlighted area ratio is unusually large

### System behavior
- Reliability confidence is penalized due to low spatial specificity
- Final label after triage: uncertain

### Interpretation
Some defects affect large regions or interact with background patterns.
In such cases:
- anomaly evidence is spread out,
- heatmap localization becomes weak,
- automated interpretation is less reliable.

The uncertain label reflects **model uncertainty**, not defect absence.

### Operational guidance
Manual inspection is required.
The system intentionally defers the final decision to a human operator.

---

## Key takeaway

The **uncertain** label means:

- The system detected atypical patterns
- But confidence is insufficient for automation
- Human review is required before action

Uncertain does **not** mean:
- false positive
- false negative
- or system error

It represents a **designed safety margin** in decision-making.