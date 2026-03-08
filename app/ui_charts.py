"""Chart-building helpers (Altair, Matplotlib) for the Streamlit viewer. Return chart/fig; caller renders with st."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import altair as alt
    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False

from app.ui_data import REASON_ORDER, get_final_label, get_reasons


def altair_dark_theme(chart: "alt.Chart") -> "alt.Chart":
    """Apply transparent background and light axis colors for Streamlit dark theme."""
    return chart.configure(background="transparent").configure_view(strokeOpacity=0).configure_axis(
        labelColor="#b0b0b0",
        titleColor="#b0b0b0",
        gridColor="#555555",
    )


def label_distribution_chart_altair(counts: Dict[str, int], height: int = 180):
    """Bar chart: order normal → uncertain → anomaly; category labels horizontal under each bar."""
    if not HAS_ALTAIR:
        return None
    order = ["normal", "uncertain", "anomaly"]
    data = [{"label": k, "samples": counts.get(k, 0)} for k in order]
    return alt.Chart(alt.Data(values=data)).mark_bar(color="#8ec1ff").encode(
        x=alt.X(
            "label",
            type="ordinal",
            sort=order,
            title="",
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y("samples", type="quantitative", title="samples"),
    ).properties(height=height)


def reason_breakdown_chart_altair(reason_hit_rates: Dict[str, float], height: int = 180):
    """Bar chart: hit-rate per reason (0–100%), style matches label distribution. Tooltip explains multi-label."""
    if not HAS_ALTAIR:
        return None
    note = "Percent of uncertain samples that include each reason (multi-label; bars do not sum to 100%)."
    data = [
        {
            "reason": r,
            "percent": float(round(reason_hit_rates.get(r, 0) * 100, 0)),
            "note": note,
        }
        for r in REASON_ORDER
    ]
    bars = (
        alt.Chart(alt.Data(values=data))
        .mark_bar(color="#8ec1ff")
        .encode(
            x=alt.X(
                "reason:N",
                sort=REASON_ORDER,
                title="",
                axis=alt.Axis(labelAngle=0, labelColor="#ddd"),
            ),
            y=alt.Y("percent:Q", title="percent", scale=alt.Scale(domain=[0, 100])),
            tooltip=[
                alt.Tooltip("reason:N", title="reason"),
                alt.Tooltip("percent:Q", title="percent", format=".0f"),
                alt.Tooltip("note:N", title="Note"),
            ],
        )
        .properties(height=height)
    )
    text = (
        alt.Chart(alt.Data(values=data))
        .transform_calculate(
            pct_label="format(datum.percent, '.0f') + '%'"
        )
        .mark_text(align="center", baseline="bottom", dy=-4, color="#ddd", size=10)
        .encode(
            x=alt.X("reason:N", sort=REASON_ORDER),
            y=alt.Y("percent:Q"),
            text="pct_label:N",
        )
    )
    return alt.layer(bars, text)


def scatter_score_confidence_altair(
    rows: List[Dict[str, Any]],
    threshold: Optional[float],
    margin_eps: float,
    conf_eps: float,
    pad: float = 0.02,
    selected_sample_id: Optional[str] = None,
):
    """Build Altair scatter: score vs confidence, color by label, threshold/conf_eps/margin band, optional current-sample highlight."""
    if not HAS_ALTAIR or not rows:
        return None
    scores = [float(r.get("score") or 0) for r in rows]
    confs = [float(r.get("confidence") or 0) for r in rows]
    labels = [get_final_label(r) for r in rows]
    reasons_list = [get_reasons(r) for r in rows]
    borderline = ["borderline_margin" in reas for reas in reasons_list]

    x_min = max(0.0, min(scores) - pad)
    x_max = min(1.0, max(scores) + pad)
    y_min = max(0.0, min(confs) - pad)
    y_max = min(1.0, max(confs) + pad)
    if threshold is not None:
        x_min = min(x_min, threshold - margin_eps - pad)
        x_max = max(x_max, threshold + margin_eps + pad)
    y_min = min(y_min, conf_eps - pad)
    y_max = max(y_max, conf_eps + pad)
    if x_min >= x_max:
        x_min, x_max = 0.0, 1.0
    if y_min >= y_max:
        y_min, y_max = 0.0, 1.0

    data = [
        {"score": s, "confidence": c, "label": lb, "borderline": b}
        for s, c, lb, b in zip(scores, confs, labels, borderline)
    ]
    scale_x = alt.Scale(domain=[x_min, x_max])
    scale_y = alt.Scale(domain=[y_min, y_max])
    color_scale = alt.Scale(domain=["normal", "uncertain", "anomaly"], range=["green", "orange", "red"])

    layers = []

    if threshold is not None:
        lo = max(0.0, threshold - margin_eps)
        hi = min(1.0, threshold + margin_eps)
        band = alt.Chart(alt.Data(values=[{"x_min": lo, "x_max": hi, "y_min": y_min, "y_max": y_max}])).mark_rect(opacity=0.15, color="gray").encode(
            x=alt.X("x_min:Q", scale=scale_x),
            x2=alt.X2("x_max:Q"),
            y=alt.Y("y_min:Q", scale=scale_y),
            y2=alt.Y2("y_max:Q"),
        )
        layers.append(band)
    if threshold is not None:
        rule_t = alt.Chart(alt.Data(values=[{"x": threshold}])).mark_rule(strokeWidth=2, color="gray").encode(
            x=alt.X("x:Q", scale=scale_x),
        )
        layers.append(rule_t)
    rule_c = alt.Chart(alt.Data(values=[{"y": conf_eps}])).mark_rule(strokeWidth=1.5, color="darkgray", strokeDash=[4, 2]).encode(
        y=alt.Y("y:Q", scale=scale_y),
    )
    layers.append(rule_c)

    points_all = alt.Chart(alt.Data(values=data)).mark_circle(size=45, opacity=0.6).encode(
        x=alt.X("score:Q", scale=scale_x, title="score"),
        y=alt.Y("confidence:Q", scale=scale_y, title="confidence"),
        color=alt.Color("label:N", scale=color_scale, legend=alt.Legend(title="")),
    )
    layers.append(points_all)
    uncertain_data = [d for d in data if d["label"] == "uncertain"]
    if uncertain_data:
        points_uncertain = alt.Chart(alt.Data(values=uncertain_data)).mark_circle(size=55, opacity=0.7, stroke="black", strokeWidth=1.2).encode(
            x=alt.X("score:Q", scale=scale_x),
            y=alt.Y("confidence:Q", scale=scale_y),
            color=alt.value("orange"),
        )
        layers.append(points_uncertain)
    borderline_data = [{"score": d["score"], "confidence": d["confidence"]} for d in data if d.get("borderline")]
    if borderline_data:
        ring = alt.Chart(alt.Data(values=borderline_data)).mark_circle(size=120, fill="none", stroke="white", strokeWidth=2).encode(
            x=alt.X("score:Q", scale=scale_x),
            y=alt.Y("confidence:Q", scale=scale_y),
        )
        layers.append(ring)

    if selected_sample_id:
        current_row = next((r for r in rows if r.get("sample_id") == selected_sample_id), None)
        if current_row is not None:
            s = float(current_row.get("score") or 0)
            c = float(current_row.get("confidence") or 0)
            current_data = [{"score": s, "confidence": c, "label": "current sample"}]
            current_scale = alt.Scale(domain=["current sample"], range=["#00e5ff"])
            current_pt = alt.Chart(alt.Data(values=current_data)).mark_point(
                shape="diamond", size=80, stroke="black", strokeWidth=2, filled=True
            ).encode(
                x=alt.X("score:Q", scale=scale_x),
                y=alt.Y("confidence:Q", scale=scale_y),
                color=alt.Color("label:N", scale=current_scale, legend=alt.Legend(title="")),
            )
            layers.append(current_pt)

    return alt.layer(*layers).properties(title="Score vs confidence", width=500, height=340)


def build_signal_gauge_figure(
    value: float,
    threshold: float,
    *,
    width: float = 6.0,
    height: float = 0.55,
):
    """Build matplotlib figure for reliability signal gauge (horizontal track, fill, threshold line). Caller uses st.pyplot(fig); plt.close(fig)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    value = min(1.0, max(0.0, float(value)))
    fig, ax = plt.subplots(figsize=(width, height), dpi=160)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    y0, h = 0.35, 0.30
    track_color = "#555555"
    fill_color = "#888888"
    ax.add_patch(Rectangle((0, y0), 1, h, facecolor=track_color, edgecolor="none"))
    ax.add_patch(Rectangle((0, y0), value, h, facecolor=fill_color, edgecolor="none", alpha=0.85))
    ax.add_patch(
        Rectangle(
            (value, y0),
            1 - value,
            h,
            facecolor=track_color,
            alpha=0.35,
            hatch="....",
            edgecolor="#666",
        )
    )
    ax.axvline(threshold, ymin=0.25, ymax=0.75, color="red", linewidth=2, alpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig
