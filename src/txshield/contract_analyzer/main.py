"""Public entry point for the paper-aligned transaction analysis pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from txshield.contract_analyzer.backends.factory import create_llm
from txshield.contract_analyzer.config import CONFIG
from txshield.contract_analyzer.core.embeddings import JinaEmbedding
from txshield.contract_analyzer.core.rag_chain import create_conversational_chain
from txshield.contract_analyzer.core.state import skip_rag_manager
from txshield.contract_analyzer.core.vector_store import load_or_create_vector_store
from txshield.contract_analyzer.models.xgb_reranker import get_xgb_reranker
from txshield.contract_analyzer.utils.monitor import SimplePromptMonitor


def analyze_contract(
    query: str,
    system_input_path: str | None = None,
    system_input_text: str | None = None,
) -> dict[str, Any]:
    """Analyze one transaction while preserving the original public interface."""
    llm_response: Any = None
    try:
        skip_rag_manager.reset()
        embedding = JinaEmbedding()
        vector_store = load_or_create_vector_store(CONFIG["PERSIST_DIR"], embedding)
        llm = create_llm()

        if system_input_path is not None:
            system_input = load_system_input(system_input_path)
        elif system_input_text is not None:
            system_input = system_input_text
        else:
            system_input = ""

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
        llm_response = chain.invoke(
            input_data,
            config={"callbacks": [SimplePromptMonitor()]},
        )

        if isinstance(llm_response, dict) and "answer" in llm_response:
            result = llm_response
            result.setdefault("used_rag", not skip_rag_manager.get_skip_rag())
            result.setdefault("similarity_info", None)
            return result

        if hasattr(llm_response, "additional_kwargs"):
            result = {
                "answer": llm_response.content,
                "used_rag": not skip_rag_manager.get_skip_rag(),
                "similarity_info": None,
                "xgb_prob": getattr(llm_response, "xgb_prob", None),
            }
            for key in ("_full_prompt_content", "_api_messages"):
                if key in llm_response.additional_kwargs:
                    result[key] = llm_response.additional_kwargs[key]
            return result

        return {
            "answer": f"Unexpected response type: {type(llm_response)}",
            "used_rag": False,
            "similarity_info": None,
            "error": "Invalid response object",
        }
    except Exception as exc:
        return {
            "error": f"Analysis failed: {exc}",
            "status": "failed",
            "similarity_info": None,
            "used_rag": False,
            "xgb_prob": (
                llm_response.get("xgb_prob")
                if isinstance(llm_response, dict)
                else None
            ),
        }


def load_system_input(file_path: str) -> str:
    """Load and normalize a transaction JSON file for the legacy path API."""
    try:
        with Path(file_path).open("r", encoding="utf-8") as source:
            return json.dumps(json.load(source), ensure_ascii=False)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ""
