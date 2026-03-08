"""Anomaly Detection Monitoring Dashboard: run-level monitoring + sample detail."""
from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from app import ui_data
from app import ui_sections


def _bump_on_run_dir_change(run_dir: str) -> int:
    """Bump label-dist nonce when run_dir changes; clear caches so chart and data refresh."""
    prev = st.session_state.get("_prev_run_dir")
    if prev != run_dir:
        st.session_state["_prev_run_dir"] = run_dir
        st.session_state["_label_dist_nonce"] = int(st.session_state.get("_label_dist_nonce", 0)) + 1
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            st.cache_resource.clear()
        except Exception:
            pass
    return int(st.session_state.get("_label_dist_nonce", 0))


st.set_page_config(page_title="Anomaly Detection Monitoring Dashboard", layout="wide")
st.title("Anomaly Detection Monitoring Dashboard")

runs = ui_data.discover_runs()
query_run = st.query_params.get("run", "").strip() if hasattr(st, "query_params") else ""
default_index = 0
if runs:
    for i, r in enumerate(runs):
        if r["display"] == query_run:
            default_index = i
            break
    selected_run = st.sidebar.selectbox(
        "Run",
        options=runs,
        format_func=lambda r: r["display"],
        index=default_index,
        key="run_select",
    )
    run_dir_str = selected_run["run_dir"]
    if st.sidebar.checkbox("Custom run path", value=False, key="use_custom_run"):
        with st.sidebar.expander("Custom run path", expanded=False):
            run_dir_str = st.text_input("run_dir", value=run_dir_str, key="run_dir_custom")
else:
    run_dir_str = st.sidebar.text_input("run_dir", value="artifacts/runs", key="run_dir_custom")
run_dir = Path(run_dir_str)
if not run_dir.exists():
    st.warning(f"Run directory not found: {run_dir}")
    st.info("Generate artifacts first using the CLI.")
    st.code(
        "python -m src.cli.run --input inputs/ --out artifacts/runs --run_id demo_run",
        language="bash",
    )
    st.stop()
label_dist_nonce = _bump_on_run_dir_change(str(run_dir))

if "run_dir_prev" not in st.session_state:
    st.session_state.run_dir_prev = None
if "chart_nonce" not in st.session_state:
    st.session_state.chart_nonce = 0
if st.session_state.run_dir_prev != str(run_dir):
    st.session_state.chart_nonce += 1
    st.session_state.run_dir_prev = str(run_dir)

rows = ui_data.load_run_results(run_dir)
st.sidebar.write(f"index rows: {len(rows)}")

if rows:
    row_ids = [r.get("sample_id") for r in rows if r.get("sample_id")]
    if "selected_sample_id" not in st.session_state:
        st.session_state["selected_sample_id"] = row_ids[0] if row_ids else None
    elif st.session_state.get("selected_sample_id") not in row_ids:
        st.session_state["selected_sample_id"] = row_ids[0] if row_ids else None

ui_sections.render_monitoring_panel(rows, run_dir, label_dist_nonce, st.session_state.chart_nonce)

exclude_na_null = st.sidebar.checkbox("Exclude NA/null", value=False, help="Exclude samples where evaluation.error_type is NA or null from the list and filter options.")
if exclude_na_null:
    rows_for_filter = [r for r in rows if r.get("error_type") not in (None, "NA")]
else:
    rows_for_filter = rows

error_type_options = sorted({(r.get("error_type") or "NA") for r in rows})
default_selected = [x for x in error_type_options if x not in ("NA",)] if exclude_na_null else error_type_options
selected_error = st.sidebar.multiselect("Filter: evaluation.error_type", options=error_type_options, default=default_selected)

query = st.sidebar.text_input("Search (sample_id / filename)")

filtered = []
for r in rows_for_filter:
    et = r.get("error_type") or "NA"
    if selected_error and et not in selected_error:
        continue
    q = (query or "").strip().lower()
    if q:
        hay = f"{r.get('sample_id','')} {r.get('input_filename','')}".lower()
        if q not in hay:
            continue
    filtered.append(r)

st.subheader("Samples")
st.write(f"Showing {len(filtered)} / {len(rows)}")

sample_ids = [r.get("sample_id") for r in filtered if r.get("sample_id")]
current_id = st.session_state.get("selected_sample_id")
if current_id and current_id not in sample_ids and any(r.get("sample_id") == current_id for r in rows):
    sample_ids = [current_id] + [s for s in sample_ids if s != current_id]
if sample_ids:
    selected_sample = st.selectbox("Select sample_id", options=sample_ids, key="selected_sample_id")
else:
    selected_sample = None

if not selected_sample:
    st.info("No samples found. Run the CLI first to generate artifacts.")
    st.stop()

row = next(r for r in filtered if r["sample_id"] == selected_sample)
result = ui_data.load_result(run_dir, row["result_relpath"])
ui_sections.render_sample_detail(run_dir, row, result)
