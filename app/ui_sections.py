"""Section-level Streamlit rendering: run-level monitoring, sample detail, reliability, LLM, summary."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from app import ui_charts
from app import ui_data
from app import ui_llm
from src.pipeline.triage import (
    DEFAULT_AREA_HI,
    DEFAULT_CONCENTRATION_LO,
    DEFAULT_CONF_EPS,
    DEFAULT_MARGIN_EPS,
)


def render_monitoring_panel(
    rows: List[Dict[str, Any]],
    run_dir: Path,
    label_dist_nonce: int,
    chart_nonce: int,
) -> None:
    """Run-level metrics, operational summary, label chart, reason breakdown, scatter, top uncertain, drift hint."""
    if not rows:
        return
    threshold, mode = ui_data.get_run_threshold_info(run_dir)
    stats = ui_data.compute_run_stats(rows)
    total = stats["total_samples"]
    counts = stats["label_counts"]
    u_ratio = stats["uncertain_ratio"]
    a_ratio = stats["anomaly_ratio"]

    st.subheader("Run-level monitoring")
    st.caption("This run prioritizes recall; high uncertain ratio indicates conservative triage under limited calibration.")

    base_anomaly_count = sum(1 for r in rows if ui_data.get_base_label(r, threshold) == "anomaly")
    confident_anomaly_count = sum(
        1 for r in rows
        if ui_data.get_base_label(r, threshold) == "anomaly" and len(ui_data.get_reasons(r)) == 0
    )
    confident_anomaly_rate = (confident_anomaly_count / base_anomaly_count * 100) if base_anomaly_count > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total samples", total)
    with c2:
        st.metric("Uncertain %", f"{u_ratio * 100:.1f}%", help=("Samples flagged by triage as borderline or low-trust.\n"
                                                                "These cases are routed for human review."))
    with c3:
        st.metric("Anomaly %", f"{a_ratio * 100:.1f}%")
    with c4:
        st.metric("Confident anomaly %", f"{confident_anomaly_rate:.1f}%", help="Portion of model-proposed anomalies that pass triage and are considered automation-ready.")

    chart_key = f"label_dist::{run_dir}::{label_dist_nonce}"
    label_dist_ph = st.empty()
    with label_dist_ph:
        st.caption(f"debug label_dist nonce={label_dist_nonce} run_dir={run_dir} counts={counts}")
        st.markdown("**Label distribution**")
        if ui_charts.HAS_ALTAIR:
            chart = ui_charts.label_distribution_chart_altair(counts)
            if chart is not None:
                spec = chart.to_dict()
                spec.setdefault("usermeta", {})
                spec["usermeta"]["nonce"] = label_dist_nonce
                spec["usermeta"]["run_dir"] = str(run_dir)
                st.vega_lite_chart(spec=spec, use_container_width=True, key=chart_key)
        else:
            import matplotlib.pyplot as plt
            order = ["normal", "uncertain", "anomaly"]
            samples = [counts.get(k, 0) for k in order]
            fig, ax = plt.subplots(figsize=(6, 2.5), dpi=100)
            ax.bar(order, samples, color="#8ec1ff")
            ax.set_ylabel("samples")
            ax.set_xticklabels(order, rotation=0)
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    uncertain_rows = [r for r in rows if ui_data.get_final_label(r) == "uncertain"]
    st.markdown("**Reason breakdown (uncertain only)**")
    if not uncertain_rows:
        st.write("No uncertain samples in this run.")
    else:
        rates = ui_data.reason_hit_rates(uncertain_rows)
        chart = ui_charts.reason_breakdown_chart_altair(rates)
        if chart is not None:
            chart = ui_charts.altair_dark_theme(chart)
            st.altair_chart(
                chart,
                use_container_width=True,
                key=f"reason_breakdown_{chart_nonce}",
            )
        drift_text = ui_data.drift_hint_text(rates)
        if drift_text:
            st.caption(drift_text)

    selected_sample_id = st.session_state.get("selected_sample_id")
    scatter_chart = ui_charts.scatter_score_confidence_altair(
        rows, threshold, DEFAULT_MARGIN_EPS, DEFAULT_CONF_EPS, selected_sample_id=selected_sample_id
    )
    if scatter_chart is not None:
        scatter_chart = ui_charts.altair_dark_theme(scatter_chart)
        st.altair_chart(scatter_chart, use_container_width=True)
    mode_str = (mode or "unknown").strip() or "unknown"
    cap = f"Threshold={threshold:.4f} | mode={mode_str} | margin_eps={DEFAULT_MARGIN_EPS} | conf_eps={DEFAULT_CONF_EPS}"
    if threshold is None:
        cap = f"Threshold=— | mode={mode_str} | margin_eps={DEFAULT_MARGIN_EPS} | conf_eps={DEFAULT_CONF_EPS}"
    st.caption(cap)

    st.markdown("**Top uncertain samples to review**")
    top_list = ui_data.top_uncertain_rows(rows, threshold, n=3)
    if top_list:
        try:
            import pandas as pd
            top_df = pd.DataFrame(top_list)
        except ImportError:
            top_df = top_list
        event = st.dataframe(
            top_df,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="top_uncertain_table",
        )
        if event and getattr(event, "selection", None) and getattr(event.selection, "rows", None) and event.selection.rows:
            idx = event.selection.rows[0]
            st.session_state["selected_sample_id"] = top_list[idx]["sample_id"]
    else:
        st.caption("No uncertain samples.")


def render_sample_detail(run_dir: Path, row: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Sample detail: images, Result header, Triage table, reliability gauges, LLM block, Summary dict, full result expander."""
    import matplotlib.pyplot as plt

    sample_id = row.get("sample_id") or ""
    sample_dir = run_dir / "samples" / sample_id
    paths = result.get("paths", {})
    original_path = sample_dir / (paths.get("original") or "original.png")
    overlay_path = sample_dir / (paths.get("overlay") or "overlay.png")

    pred = result.get("prediction", {})
    rel = result.get("reliability", {})
    eval_ = result.get("evaluation", {})
    triage = result.get("triage", {})

    col1, col2 = st.columns(2)
    with col1:
        st.image(str(original_path), caption="original", width="stretch")
    with col2:
        st.image(str(overlay_path), caption="overlay", width="stretch")

    st.subheader("Result")
    final_label = pred.get("label") or triage.get("final_label")
    if final_label == "uncertain":
        st.warning("Label: **uncertain** (borderline or low-trust; review recommended)")
    else:
        st.write(f"**Label:** {final_label}")

    if triage:
        st.subheader("Triage")
        base_label = triage.get("base_label")
        triage_final = triage.get("final_label")
        reasons = triage.get("reasons", [])
        reasons_str = ", ".join(reasons) if reasons else "—"
        st.markdown(
            "| Field | Value |\n"
            "|-------|-------|\n"
            f"| **base_label** | {base_label or '—'} |\n"
            f"| **final_label** | {triage_final or '—'} |\n"
            f"| **reasons** | {reasons_str} |"
        )
    else:
        st.caption("_No triage section (older run)._")

    rel_signals = rel.get("signals") or {}
    area_ratio = rel_signals.get("area_ratio")
    heatmap_concentration = rel_signals.get("heatmap_concentration")
    if area_ratio is not None or heatmap_concentration is not None:
        st.caption("_Reliability signals (sample)_")
        col1, spacer, col2 = st.columns([1, 0.15, 1])
        with col1:
            if area_ratio is not None:
                ar = float(area_ratio)
                area_threshold = DEFAULT_AREA_HI
                area_status = "too large" if ar > area_threshold else "ok"
                st.markdown(
                    f'**Area ratio** <span style="color:#888;font-size:0.85em;">(threshold: {area_threshold:.2f})</span>',
                    unsafe_allow_html=True,
                )
                fig = ui_charts.build_signal_gauge_figure(ar, area_threshold)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.markdown(
                    f"<span style='color:#AAA;font-size:0.9em;'>value: {ar:.2f} · {area_status}</span>",
                    unsafe_allow_html=True,
                )
        with spacer:
            st.write("")
        with col2:
            if heatmap_concentration is not None:
                conc = float(heatmap_concentration)
                conc_threshold = DEFAULT_CONCENTRATION_LO
                conc_status = "diffuse" if conc < conc_threshold else "localized"
                st.markdown(
                    f'**Heatmap concentration** <span style="color:#888;font-size:0.85em;">(threshold: {conc_threshold:.2f})</span>',
                    unsafe_allow_html=True,
                )
                fig = ui_charts.build_signal_gauge_figure(conc, conc_threshold)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.markdown(
                    f"<span style='color:#AAA;font-size:0.9em;'>value: {conc:.2f} · {conc_status}</span>",
                    unsafe_allow_html=True,
                )

    llm = result.get("llm", {})
    if final_label == "uncertain" and (llm.get("summary") or llm.get("explanation")):
        st.subheader("LLM (uncertain review)")
        if llm.get("summary"):
            st.write("**Summary**")
            st.write(llm["summary"])
        if llm.get("explanation"):
            st.write("**Explanation**")
            expl = llm["explanation"]
            if "(based on " in expl:
                ui_llm.render_explanation_with_secondary_attribution(expl)
            else:
                st.markdown(expl)
        if llm.get("notes"):
            st.caption(llm["notes"])
    elif llm.get("notes") == "skipped_non_uncertain":
        st.caption("_LLM skipped (non-uncertain sample)._")

    st.subheader("Summary")
    summary = {
        "sample_id": result.get("sample_id"),
        "label": final_label,
        "score": pred.get("score"),
        "threshold": pred.get("threshold"),
        "confidence": rel.get("confidence"),
        "error_type": eval_.get("error_type"),
        "input_filename": result.get("meta", {}).get("input_filename"),
        "created_at": result.get("created_at"),
    }
    st.write(summary)

    with st.expander("Show full result.json"):
        st.json(result)
