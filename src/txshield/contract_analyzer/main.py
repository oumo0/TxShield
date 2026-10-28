from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from txshield.common.trace_utils import validate_trace_text
from txshield.contract_analyzer.backends.factory import create_llm
from txshield.contract_analyzer.config import CONFIG
from txshield.contract_analyzer.core.embeddings import JinaEmbedding
from txshield.contract_analyzer.core.rag_chain import create_conversational_chain
from txshield.contract_analyzer.core.state import skip_rag_manager
from txshield.contract_analyzer.core.vector_store import load_or_create_vector_store
from txshield.contract_analyzer.models.xgb_reranker import get_xgb_reranker
from txshield.contract_analyzer.pipeline.decision import apply_llm_decision


def _resolve_system_input(
    system_input_path: str | None,
    system_input_text: str | None,
) -> str:
    if system_input_path is not None:
        return load_system_input(system_input_path)
    return system_input_text or ""


def _normalize_response(response: Any) -> dict[str, Any]:
    used_rag = not skip_rag_manager.get_skip_rag()
    if isinstance(response, dict) and "answer" in response:
        response.setdefault("used_rag", used_rag)
        response.setdefault("similarity_info", None)
        if "llm_prediction" not in response:
            return apply_llm_decision(
                response,
                xgb_probability=response.get("xgb_prob"),
                context_threshold=response.get("xgb_threshold"),
                context_retained=response.get("xgb_context_retained"),
            )
        return response

    if hasattr(response, "additional_kwargs") and hasattr(response, "content"):
        metadata = response.additional_kwargs
        result: dict[str, Any] = {
            "answer": response.content,
            "used_rag": used_rag,
            "similarity_info": None,
        }
        for key in ("_full_prompt_content", "_api_messages"):
            if key in metadata:
                result[key] = metadata[key]
        return apply_llm_decision(
            result,
            xgb_probability=getattr(response, "xgb_prob", None),
            context_threshold=None,
            context_retained=None,
        )

    raise TypeError(f"Unexpected response type: {type(response).__name__}")


def _failure_result(exc: Exception, response: Any) -> dict[str, Any]:
    probability = response.get("xgb_prob") if isinstance(response, dict) else None
    return {
        "answer": "",
        "error": f"Analysis failed ({type(exc).__name__})",
        "status": "failed",
        "llm_prediction": "Unknown",
        "final_prediction": "Unknown",
        "similarity_info": None,
        "used_rag": False,
        "xgb_prob": probability,
        "xgb_attack_probability": probability,
        "xgb_applied": probability is not None,
    }


def analyze_contract(
    query: str,
    system_input_path: str | None = None,
    system_input_text: str | None = None,
) -> dict[str, Any]:
    """Analyze one transaction while preserving the original public interface."""
    llm_response: Any = None
    try:
        skip_rag_manager.reset()
        system_input = _resolve_system_input(system_input_path, system_input_text)
        validate_trace_text(system_input)
        embedding = JinaEmbedding()
        vector_store = load_or_create_vector_store(CONFIG["PERSIST_DIR"], embedding)
        llm = create_llm()

        input_data = {
            "input": query,
            "system_input": system_input,
            "chat_history": [],
            "skip_rag": False,
        }
        chain = create_conversational_chain(
            vector_store,
            llm,
            csv_file_path=CONFIG["CSV_PATH"],
            original_input=input_data,
            xgb_reranker=get_xgb_reranker(),
        )
        llm_response = chain.invoke(input_data)
        return _normalize_response(llm_response)
    except Exception as exc:
        return _failure_result(exc, llm_response)


def load_system_input(file_path: str) -> str:
    """Load and normalize a transaction JSON file for the legacy path API."""
    with Path(file_path).open("r", encoding="utf-8") as source:
        return json.dumps(json.load(source), ensure_ascii=False)
