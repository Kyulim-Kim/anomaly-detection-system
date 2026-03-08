# Anomaly Detection System – Internal Operations Guide

This document describes the internal operational guidelines for an image-based
anomaly detection system used as a **decision support tool**.

This system does NOT make final production decisions.
Its outputs are designed to:
- Highlight suspicious or atypical samples
- Assist operators in prioritizing inspection effort
- Provide visual and contextual signals to support human judgment

Final responsibility always lies with a human operator.

---

## System outputs and decision layers

The system produces the following key outputs:

- **Decision score (anomaly score)**  
  A continuous measure of how atypical a sample appears relative to normal data.

- **Threshold-based base label (normal / anomaly)**  
  A first-stage decision derived purely from the anomaly score and a fixed threshold.

- **Reliability confidence**  
  An operational signal indicating how trustworthy or stable the decision is.  
  This is **not** a probability of defect and is not used as a standalone decision variable.

- **Uncertain triage (second-stage override)**  
  A policy-driven override applied when confidence is low or signals are ambiguous.
  The uncertain label represents an **intentional low-trust region** that requires human review.

- **Visual heatmap and detected regions**  
  Spatial cues showing where anomaly evidence is relatively stronger,
  used to support inspection rather than confirm defects.

---

## Operational design principles

This system is intentionally designed to favor **conservative operation**:

- Automation is applied only when decision confidence is sufficient
- Ambiguous cases are explicitly surfaced via the *uncertain* label
- Human-in-the-loop review is treated as a first-class design component,
  not an exception or fallback

The goal is to reduce inspection workload **without compromising decision safety**.