# Heatmap Interpretation Guidelines

The heatmap visualizes **pixel-level anomaly evidence distribution**.
It describes **where and how anomaly evidence is expressed spatially**,
not whether a defect exists.

Important:
- Heatmap patterns support human interpretation.
- They must not be used as standalone defect decisions.
- Localized ≠ defect
- Diffuse ≠ normal

## Key interpretation principles

- **Localized, relatively high-intensity regions** indicate spatially concentrated anomaly evidence.
  This evidence may arise from true defects, but also from normal structural patterns,
  reflections, or repetitive textures.

- **Diffuse heatmaps** indicate broadly distributed or weak anomaly evidence.
  This often appears when decisions are near the threshold,
  image quality is inconsistent, or global patterns influence the model response.

- **Very large highlighted areas** suggest low spatial specificity of evidence in many cases.
  Such patterns may be influenced by background shift, illumination changes,
  or texture variation, and are commonly treated as **lower-confidence signals**.

## Hotspots

Hotspots represent connected regions exceeding a **visualization threshold**
used to summarize relatively stronger anomaly evidence.

Only top-ranked regions are shown to improve interpretability.

The presence of hotspots does **not** confirm a defect.
They indicate where anomaly evidence is comparatively stronger within the image.

## What operators should focus on

When reviewing heatmaps, operators should consider:
- Spatial distribution of evidence (localized vs diffuse)
- Relative strength and consistency of highlighted regions
- Visual alignment with known defect characteristics
- Overall image quality (lighting, focus, reflections, background)

Heatmap interpretation should always be combined with:
- Score-to-threshold distance
- Reliability confidence signals
- Triage outcomes (e.g., uncertain decisions)