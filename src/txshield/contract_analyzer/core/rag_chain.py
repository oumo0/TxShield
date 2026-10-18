"""Paper-aligned retrieval, LLM inference, and XGBoost post-filter chain."""
from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda

from txshield.contract_analyzer.config import CONFIG
from txshield.contract_analyzer.core.state import skip_rag_manager
from txshield.contract_analyzer.models.xgb_reranker import XGBoostReranker
from txshield.contract_analyzer.pipeline.decision import apply_post_filter


def _replace_ids(value: Any, descriptions: dict[str, str]) -> Any:
    """Replace historical function IDs with their security descriptions."""
    if isinstance(value, list):
        return [_replace_ids(item, descriptions) for item in value]
    if not isinstance(value, dict):
        return value

    replaced: dict[str, Any] = {}
    for key, item in value.items():
        if key == "id":
            description = descriptions.get(str(item))
            if description is not None:
                replaced["description_Security"] = description
            continue
        replaced[key] = _replace_ids(item, descriptions)
    return replaced


def _load_descriptions(csv_file_path: str) -> dict[str, str]:
    frame = pd.read_csv(
        csv_file_path,
        usecols=["id", "description_Security"],
        encoding="MacRoman",
    )
    return dict(zip(frame["id"].astype(str), frame["description_Security"].astype(str)))


def _build_reference_text(
    documents: list[Document], descriptions: dict[str, str]
) -> str:
    """Serialize retrieved cases, preserving their execution-flow evidence."""
    cases: list[str] = []
    for index, document in enumerate(documents, start=1):
        try:
            content = json.dumps(
                _replace_ids(json.loads(document.page_content), descriptions),
                ensure_ascii=False,
            )
        except (TypeError, json.JSONDecodeError):
            content = document.page_content
        distance = document.metadata.get("similarity_score") if document.metadata else None
        heading = f"Historical case {index}"
        if distance is not None:
            heading += f" (distance={float(distance):.6f})"
        cases.append(f"{heading}:\n{content}")
    return "\n\n".join(cases)


def _similarity_info(documents: list[Document]) -> dict[str, Any] | None:
    scores = [
        float(document.metadata["similarity_score"])
        for document in documents
        if document.metadata and "similarity_score" in document.metadata
    ]
    if not scores:
        return None
    return {
        "min_score": min(scores),
        "max_score": max(scores),
        "avg_score": sum(scores) / len(scores),
        "doc_count": len(scores),
        "all_scores": scores,
    }


def create_conversational_chain(
    vector_store: Chroma,
    llm: Runnable,
    csv_file_path: str,
    original_input: dict[str, Any],
    xgb_reranker: XGBoostReranker | None = None,
) -> Runnable:
    """Create the pipeline while ensuring XGBoost cannot gate retrieval or LLM input."""

    def retrieve(input_data: dict[str, Any]) -> tuple[list[Document], str | None]:
        if input_data.get("skip_rag", False):
            skip_rag_manager.set_skip_rag(True)
            return [], None

        transaction = str(
            input_data.get("system_input")
            or original_input.get("system_input")
            or input_data.get("input")
            or original_input.get("input")
            or ""
        )
        if not transaction:
            skip_rag_manager.set_skip_rag(True)
            return [], "No transaction text was available for retrieval"

        try:
            candidates = vector_store.similarity_search_with_score(
                transaction,
                k=int(os.getenv("TXSHIELD_RETRIEVAL_K", "5")),
            )
        except Exception as exc:
            skip_rag_manager.set_skip_rag(True)
            return [], f"Retrieval failed: {exc}"

        threshold = CONFIG["SIMILARITY_THRESHOLD"]
        documents: list[Document] = []
        for document, score in candidates:
            distance = float(score)
            if distance > threshold:
                continue
            document.metadata = dict(document.metadata or {})
            document.metadata["similarity_score"] = distance
            document.metadata["score"] = distance
            documents.append(document)

        skip_rag_manager.set_skip_rag(not documents)
        input_data["skip_rag"] = not documents
        input_data["_retrieved_documents"] = documents
        return documents, None

    def run_pipeline(input_data: dict[str, Any]) -> dict[str, Any]:
        documents, retrieval_error = retrieve(input_data)
        reference_text = ""
        if documents:
            try:
                reference_text = _build_reference_text(
                    documents,
                    _load_descriptions(csv_file_path),
                )
            except (OSError, ValueError, KeyError) as exc:
                retrieval_error = f"Knowledge metadata failed: {exc}"
                skip_rag_manager.set_skip_rag(True)

        used_rag = bool(documents and reference_text and not retrieval_error)
        if not used_rag:
            reference_text = ""
            skip_rag_manager.set_skip_rag(True)

        llm_input = {
            "query": input_data.get("input", original_input.get("input", "")),
            "system_input": input_data.get(
                "system_input", original_input.get("system_input", "")
            ),
            "reference_text": reference_text,
            "skip_rag": not used_rag,
            "prompt_type": "rag" if used_rag else "direct",
        }
        llm_response = llm.invoke(llm_input)
        result: dict[str, Any] = {
            "answer": (
                llm_response.content
                if hasattr(llm_response, "content")
                else str(llm_response)
            ),
            "context": documents,
            "used_rag": used_rag,
            "similarity_info": _similarity_info(documents),
        }
        if retrieval_error:
            result["retrieval_error"] = retrieval_error
        if "_retrieved_documents" in input_data:
            result["_retrieved_documents"] = input_data["_retrieved_documents"]

        return apply_post_filter(
            result,
            str(input_data.get("system_input", "")),
            xgb_reranker,
        )

    return RunnableLambda(run_pipeline)
