"""LLM explanation parsing and rendering for the Streamlit viewer."""
from __future__ import annotations

import re
from typing import List

import streamlit as st


def extract_attribution_and_clean_expl(explanation: str) -> tuple[str, str]:
    """Extract a single de-duplicated attribution string and cleaned explanation without per-section (based on ...)."""
    pattern = re.compile(r"\(based on ([^)]+)\)", re.IGNORECASE)
    seen: set[str] = set()
    order: List[str] = []
    for m in pattern.finditer(explanation):
        part = m.group(1).strip()
        for token in (p.strip() for p in part.split(",")):
            if token and token not in seen:
                seen.add(token)
                order.append(token)
    attrib_str = ", ".join(order) if order else ""
    cleaned = pattern.sub("", explanation)
    cleaned = re.sub(r"\s*:\s*(\n|$)", r"\1", cleaned)
    cleaned = re.sub(r"  +", " ", cleaned).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return attrib_str, cleaned


def render_explanation_with_secondary_attribution(explanation: str) -> None:
    """Render explanation with attribution shown once under the Explanation header; body has (based on ...) removed from headings."""
    attrib_str, cleaned = extract_attribution_and_clean_expl(explanation)
    if attrib_str:
        st.caption(f"(based on {attrib_str})")
    st.markdown(cleaned)
