"""RAG (MVP) + LLM. RAG uses LocalFolderStore + KeywordRetriever. LLM: OpenAI SDK or vLLM-compatible or stub."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag.rag_pipeline import RAGEngine
from src.rag.retriever import KeywordRetriever
from src.rag.store import LocalFolderStore

# Default context length limits
MAX_CONTEXT_CHARS_RESULT = 1500
MAX_CONTEXT_CHARS_LLM = 3500


@dataclass(frozen=True)
class RagOutput:
    context_used: bool
    contexts: List[Dict[str, Any]]  # = contexts_for_result (for result.json)
    contexts_for_llm: List[Dict[str, Any]]  # for LLM input (longer text)
    notes: str = ""


@dataclass(frozen=True)
class LlmOutput:
    summary: str
    explanation: str
    notes: str = ""


def run_rag(
    query: str,
    *,
    docs_dir: str = "assets/knowledge",
    top_k: int = 5,
    force_rebuild: bool = False,
    cache_path: Path = Path("artifacts/knowledge_index.jsonl"),
    max_chars_result: int = MAX_CONTEXT_CHARS_RESULT,
    max_chars_llm: int = MAX_CONTEXT_CHARS_LLM,
) -> RagOutput:
    """MVP RAG: LocalFolderStore + KeywordRetriever + RAGEngine. Returns contexts_for_result (rag) and contexts_for_llm (LLM)."""
    root = Path(docs_dir)
    if not root.exists():
        return RagOutput(
            context_used=False,
            contexts=[],
            contexts_for_llm=[],
            notes=f"Docs dir not found: {docs_dir}",
        )
    store = LocalFolderStore(root, glob="**/*.md")
    retriever = KeywordRetriever()
    engine = RAGEngine(store=store, retriever=retriever, cache_path=cache_path)
    index_status = engine.build_or_load_index(force_rebuild=force_rebuild)
    raw = engine.retrieve(
        query,
        top_k=top_k,
        max_chars_result=max_chars_result,
        max_chars_llm=max_chars_llm,
    )
    notes = raw.get("notes", "")
    if index_status.get("cache_invalidated"):
        notes = (notes + " cache_invalidated=true").strip()
    docs_count = index_status.get("docs_count")
    chunks_count = index_status.get("chunks_count")
    if docs_count is not None and chunks_count is not None:
        notes = (notes + f" docs_count={docs_count} chunks={chunks_count}").strip()
    if getattr(store, "get_repo_root_marker", lambda: "")() == "README.md":
        notes = (notes + " repo_root_fallback=README").strip()
    return RagOutput(
        context_used=raw["context_used"],
        contexts=raw["contexts_for_result"],
        contexts_for_llm=raw["contexts_for_llm"],
        notes=notes,
    )


def run_rag_stub() -> RagOutput:
    return RagOutput(context_used=False, contexts=[], contexts_for_llm=[], notes="")


def run_llm_stub() -> LlmOutput:
    return LlmOutput(summary="", explanation="", notes="")


def _llm_client_and_model(timeout: float = 30.0) -> tuple[Any, str, str]:
    """Returns (OpenAI client or None, model name, provider). Same code path for OpenAI and vLLM (base_url set for vLLM)."""
    provider = (os.environ.get("LLM_PROVIDER") or os.environ.get("RAG_LLM_MODE") or "openai").strip().lower()
    if provider not in ("openai", "vllm", "stub"):
        provider = "openai"
    if provider == "stub":
        return None, "", "stub"

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, "", provider

    model = (os.environ.get("LLM_MODEL") or "gpt-4o-mini").strip()
    base_url = os.environ.get("LLM_BASE_URL") or None
    if provider == "vllm" and not base_url:
        base_url = "http://localhost:8000/v1"

    try:
        from openai import OpenAI
    except ImportError:
        return None, "", provider

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    return client, model, provider


def _classify_error(e: BaseException) -> str:
    """Classify for notes: rate_limit, server_error, timeout, connection, other."""
    err_str = str(e).lower()
    status = getattr(e, "status_code", None) or getattr(
        getattr(e, "response", None), "status_code", None
    )
    if status == 429:
        return "rate_limit"
    if status in (500, 502, 503, 504):
        return "server_error"
    if isinstance(e, TimeoutError) or "timeout" in err_str:
        return "timeout"
    if isinstance(e, ConnectionError) or "connection" in err_str or "connect" in err_str:
        return "connection"
    return "other"


def run_llm_openai(
    query: str,
    contexts: List[Dict[str, Any]],
    *,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> LlmOutput:
    """
    Single code path for OpenAI and vLLM: client.chat.completions.create(...).
    Wrapper handles retries (429, 5xx, TimeoutError, ConnectionError); max 3, exponential backoff.
    On failure: summary/explanation empty; notes include provider, error_type, retry_count.
    """
    client, actual_model, provider = _llm_client_and_model(timeout=timeout)
    if client is None or not actual_model:
        return LlmOutput(
            summary="",
            explanation="",
            notes="LLM not configured (missing key or openai package).",
        )

    ctx_lines: List[str] = []
    for c in contexts[:10]:
        title = c.get("title", "?")
        snippet = (c.get("text") or c.get("snippet", ""))[:500]
        ctx_lines.append(f"- [{title}]: {snippet}")
    context_block = "\n".join(ctx_lines) if ctx_lines else "(No retrieved documents)"

    system = (
        "You are a concise analyst for anomaly detection. "
        "Use only the provided decision context and retrieved documents. "
        "If evidence is insufficient, say so. Do not invent details."
    )
    user = (
        "Decision context:\n"
        f"{query}\n\n"
        "Retrieved documents (titles + snippets):\n"
        f"{context_block}\n\n"
        "Provide:\n"
        "1. Summary: 1-2 lines.\n"
        "2. Explanation: short bullets (evidence from heatmap + evidence from docs). "
        "If there is no relevant evidence in the docs, say 'No relevant evidence in docs.'"
    )

    last_error: Optional[str] = None
    error_type = "other"
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=actual_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=500,
                temperature=0.2,
            )
            first = (resp.choices or [None])[0]
            content = getattr(getattr(first, "message", None), "content", None) if first else None
            content = (content or "").strip() if isinstance(content, str) else ""
            if not content:
                return LlmOutput(summary="", explanation="", notes="Empty API response.")
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            summary = ""
            explanation_parts: List[str] = []
            for i, line in enumerate(lines):
                if i < 2 and not line.startswith("-") and not line.startswith("*"):
                    summary = (summary + " " + line) if summary else line
                else:
                    explanation_parts.append(line)
            explanation = "\n".join(explanation_parts) if explanation_parts else content
            if not summary:
                summary = content[:200]
            return LlmOutput(summary=summary.strip(), explanation=explanation.strip(), notes="")
        except Exception as e:
            last_error = str(e)
            error_type = _classify_error(e)
            status = getattr(e, "status_code", None) or getattr(
                getattr(e, "response", None), "status_code", None
            )
            is_retryable = (
                status in (429, 500, 502, 503, 504)
                or "timeout" in last_error.lower()
                or "connection" in last_error.lower()
            )
            if attempt < max_retries - 1 and is_retryable:
                time.sleep(2 ** attempt)
                continue
            notes = f"provider={provider} error_type={error_type} retry_count={attempt + 1}: {last_error}"
            return LlmOutput(summary="", explanation="", notes=notes)

    notes = f"provider={provider} error_type={error_type} retry_count={max_retries}: {last_error}"
    return LlmOutput(summary="", explanation="", notes=notes)


# Max length for uncertain explanation (cost + readability)
MAX_EXPLANATION_CHARS_UNCERTAIN = 1200
MAX_SUMMARY_CHARS_UNCERTAIN = 350


def _attribution_suffix(contexts: List[Dict[str, Any]]) -> str:
    """Build '(based on title1, title2)' from RAG context titles (document names only, no paths)."""
    titles = list(dict.fromkeys([c.get("title", "").strip() for c in contexts if c.get("title")]))
    if not titles:
        return ""
    return " (based on " + ", ".join(titles) + ")"


def _append_attribution_to_explanation_sections(explanation: str, attribution: str) -> str:
    """Append attribution to Why risky / What to check / Next action section headers (explanatory only)."""
    if not attribution:
        return explanation
    for header in ("**Why risky**", "**What to check**", "**Next action**"):
        if header in explanation and (header + attribution) not in explanation:
            explanation = explanation.replace(header, header + attribution, 1)
    return explanation


def _parse_uncertain_llm_response(content: str) -> tuple[str, str]:
    """Split LLM response into Summary (decision recap) and Explanation (reasoning). Summary first, then Explanation."""
    content = content.strip()
    for marker in ("**Explanation**", "**Explanation**:", "Explanation:", "Explanation\n"):
        idx = content.find(marker)
        if idx >= 0:
            summary_part = content[:idx].strip()
            explanation_part = content[idx + len(marker):].strip()
            if explanation_part.startswith(":"):
                explanation_part = explanation_part[1:].strip()
            summary_part = summary_part.replace("**Summary**", "").replace("**Summary**:", "").strip()
            summary = summary_part[:MAX_SUMMARY_CHARS_UNCERTAIN].rstrip()
            if len(summary_part) > MAX_SUMMARY_CHARS_UNCERTAIN:
                summary = summary + "..."
            return summary, explanation_part or content
    # No marker: first 2-3 lines as summary, rest as explanation
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if len(lines) <= 3:
        return content[:MAX_SUMMARY_CHARS_UNCERTAIN], content
    summary = "\n".join(lines[:3])[:MAX_SUMMARY_CHARS_UNCERTAIN].rstrip()
    explanation = "\n".join(lines[3:])
    return summary, explanation


def run_llm_uncertain(
    *,
    score: float,
    threshold: float,
    confidence: float,
    triage_reasons: List[str],
    heatmap_stats: Dict[str, Any],
    area_ratio: float,
    concentration: float,
    hotspots: List[Dict[str, Any]],
    defect_type: Optional[str],
    contexts: List[Dict[str, Any]],
    timeout: float = 30.0,
    max_retries: int = 3,
) -> LlmOutput:
    """
    LLM call for uncertain samples only. Summary = decision recap (no reasoning). Explanation = Why risky, What to check, Next action.
    Uses non-defect framing (borderline / low-confidence / human review). No raw coordinates in What to check.
    """
    client, actual_model, provider = _llm_client_and_model(timeout=timeout)
    if client is None or not actual_model:
        return LlmOutput(summary="", explanation="", notes="LLM not configured.")

    reasons_str = ", ".join(triage_reasons) if triage_reasons else "—"
    defect_str = defect_type or "(unknown)"

    ctx_lines = []
    for c in (contexts or [])[:10]:
        title = c.get("title", "?")
        snippet = (c.get("text") or c.get("snippet", ""))[:400]
        ctx_lines.append(f"- [{title}]: {snippet}")
    context_block = "\n".join(ctx_lines) if ctx_lines else "(No retrieved documents)"

    system = (
        "You are an ops analyst. This sample has final_label UNCERTAIN — that is a borderline or low-confidence decision, NOT a confirmed defect. "
        "Use terms like 'borderline decision', 'low-confidence region', 'requires human review'. "
        "Never state or imply 'defect detected' for uncertain samples. "
        "Answer based only on the provided context and internal ops documents. Do not invent details."
    )
    user = (
        "Decision context (uncertain sample; no defect confirmed):\n"
        f"- triage.reasons: {reasons_str}\n"
        f"- defect_type: {defect_str}\n\n"
        "Internal ops documents:\n"
        f"{context_block}\n\n"
        "Provide TWO sections. Do not duplicate content between them.\n\n"
        "**Summary** (2–3 bullets only; decision recap for quick scan; NO score, threshold, heatmap numbers, or 'Why risky'):\n"
        "- Decision: Uncertain\n"
        "- Nature: Borderline / low-trust decision\n"
        "- Action: Human review recommended\n"
        "Optionally add one short reason keyword (e.g. borderline_margin, low_confidence). No numeric explanation.\n\n"
        "**Explanation** (all reasoning and justification):\n"
        "1. **Why risky** (2–3 bullets): refer to reason tags; e.g. borderline score, low confidence, diffuse heatmap.\n"
        "2. **What to check** (2–3 bullets): imaging checklist (lighting, reflection, focus, background). "
        "Use visual wording only, e.g. 'Inspect highlighted regions in the heatmap overlay' or 'Focus on visually emphasized areas in the anomaly map.' "
        "Do NOT output raw coordinates or bbox values.\n"
        "3. **Next action** (1–2 bullets): e.g. re-capture, re-review, QA check."
    )

    last_error: Optional[str] = None
    error_type = "other"
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=actual_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=800,
                temperature=0.2,
            )
            first = (resp.choices or [None])[0]
            content = getattr(getattr(first, "message", None), "content", None) if first else None
            content = (content or "").strip() if isinstance(content, str) else ""
            if not content:
                return LlmOutput(summary="", explanation="", notes="Empty API response.")
            summary, explanation = _parse_uncertain_llm_response(content)
            attribution = _attribution_suffix(contexts or [])
            explanation = _append_attribution_to_explanation_sections(explanation, attribution)
            if len(explanation) > MAX_EXPLANATION_CHARS_UNCERTAIN:
                explanation = explanation[:MAX_EXPLANATION_CHARS_UNCERTAIN].rstrip() + "..."
            return LlmOutput(summary=summary, explanation=explanation, notes="")
        except Exception as e:
            last_error = str(e)
            error_type = _classify_error(e)
            status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
            is_retryable = (
                status in (429, 500, 502, 503, 504)
                or "timeout" in last_error.lower()
                or "connection" in last_error.lower()
            )
            if attempt < max_retries - 1 and is_retryable:
                time.sleep(2 ** attempt)
                continue
            notes = f"provider={provider} error_type={error_type} retry_count={attempt + 1}: {last_error}"
            return LlmOutput(summary="", explanation="", notes=notes)
    notes = f"provider={provider} error_type={error_type} retry_count={max_retries}: {last_error}"
    return LlmOutput(summary="", explanation="", notes=notes)


def run_llm(
    query: str,
    contexts: List[Dict[str, Any]],
    *,
    provider: Optional[str] = None,
) -> LlmOutput:
    """
    Single entrypoint. Provider from env: LLM_PROVIDER (or RAG_LLM_MODE) = openai | vllm | stub.
    Stub or missing key → run_llm_stub. Otherwise OpenAI SDK (with optional base_url for vLLM).
    """
    prov = (provider or os.environ.get("LLM_PROVIDER") or os.environ.get("RAG_LLM_MODE") or "openai").strip().lower()
    if prov not in ("openai", "vllm"):
        return run_llm_stub()
    client, _, _ = _llm_client_and_model()
    if client is None:
        return run_llm_stub()
    return run_llm_openai(query, contexts)
