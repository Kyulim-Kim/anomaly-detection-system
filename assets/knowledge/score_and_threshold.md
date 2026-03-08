# Score and Threshold Interpretation

This document explains how the **decision score** and **threshold**
are defined and how they should be interpreted in operation.

These values support decision-making but **do not represent defect probability**.

---

## Decision score (anomaly score)

The decision score quantifies **how atypical a sample is**
relative to the normal reference data used during calibration.

It is derived from the distribution of anomaly evidence
(e.g., anomaly map statistics), not from defect labels.

### Important characteristics

- The score is **not a probability**
- A higher score means the sample deviates more from learned normal patterns
- Scores are only comparable:
  - within the same model
  - under the same data distribution
  - using the same calibration reference

The score alone does **not** indicate whether a defect truly exists.

---

## Threshold policy

### Purpose of the threshold

The threshold defines the boundary between:
- **expected variation of normal samples**, and
- **unusual deviation requiring attention**

It is used to derive the **base decision**:
- anomaly if score ≥ threshold
- normal otherwise

### Default policy: normal_p995

By default, the threshold is computed as:

- the **99.5th percentile** of anomaly scores
- measured over a set of normal reference samples

This policy is intentionally conservative:
- Most normal samples fall below the threshold
- Only extreme deviations exceed it

Once computed, the threshold is **fixed and reused** during inference.

---

## What the threshold does (and does not do)

The threshold:
- Separates typical vs atypical behavior under normal conditions
- Controls sensitivity of the base decision

The threshold does **not**:
- Represent defect probability
- Guarantee correctness of the decision
- Account for confidence, ambiguity, or interpretability

These aspects are handled by the **triage layer**.

---

## Sensitivity trade-offs

Changing the threshold affects system behavior:

- **Lower threshold**
  - Higher sensitivity
  - More samples flagged as anomaly
  - Increased false positives

- **Higher threshold**
  - Lower sensitivity
  - Fewer anomaly detections
  - Increased false negatives

Threshold changes should be:
- Data-driven
- Logged and traceable
- Evaluated together with triage outcomes

---

## Interaction with uncertain triage

It is expected that:

- Some normal samples may exceed the threshold
- Especially when:
  - strong structural patterns exist
  - textures are repetitive
  - imaging conditions vary

Such cases are **not errors**.

They are intentionally handled by the **uncertain triage layer**,
which evaluates whether the score-based decision is trustworthy enough
for automation.

---

## Operational guidance

- Treat the score as a **relative deviation indicator**
- Treat the threshold as a **reference boundary**, not a hard truth
- Always interpret score and threshold together with:
  - reliability confidence
  - heatmap behavior
  - triage outcome

Final decisions should never rely on the score alone.

---

## Key takeaway

The decision score answers:

> “How unusual is this sample compared to normal reference data?”

It does **not** answer:

> “Is this a defect?”

That determination requires:
- triage evaluation
- visual inspection
- human judgment