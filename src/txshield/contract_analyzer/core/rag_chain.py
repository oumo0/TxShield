from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda

from txshield.contract_analyzer.config import CONFIG
from txshield.contract_analyzer.core.state import skip_rag_manager
from txshield.contract_analyzer.core.vector_store import require_l2_collection
from txshield.contract_analyzer.models.xgb_reranker import XGBoostReranker
from txshield.contract_analyzer.pipeline.decision import apply_llm_decision


@dataclass(frozen=True)
class RiskGuidance:
    probability: float | None = None
    threshold: float | None = None
    context_retained: bool | None = None
    error: str | None = None


def _replace_ids(value: Any, descriptions: dict[str, str]) -> Any:
    """Attach security descriptions while retaining historical IDs."""
    if isinstance(value, list):
        return [_replace_ids(item, descriptions) for item in value]
    if not isinstance(value, dict):
        return value

    replaced: dict[str, Any] = {}
    for key, item in value.items():
        if key == "id":
            replaced[key] = item
            description = descriptions.get(str(item))
            if description is not None:
                replaced["description_Security"] = description
            continue
        replaced[key] = _replace_ids(item, descriptions)
    return replaced


def _load_descriptions(csv_file_path: str) -> dict[str, str]:
    options = {"usecols": ["id", "description_Security"], "dtype": str, "keep_default_na": False}
    try:
        frame = pd.read_csv(csv_file_path, encoding="utf-8-sig", **options)
    except UnicodeDecodeError:
        frame = pd.read_csv(csv_file_path, encoding="MacRoman", **options)
    if frame["id"].duplicated().any() or (frame["id"] == "").any():
        raise ValueError("Knowledge IDs must be nonempty and unique")
    return dict(zip(frame["id"], frame["description_Security"]))


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
        "score_type": "l2_distance",
        "all_similarities": [1.0 - distance for distance in scores],
    }


def _transaction_text(
    input_data: dict[str, Any],
    original_input: dict[str, Any],
) -> str:
    return str(
        input_data.get("system_input")
        or original_input.get("system_input")
        or input_data.get("input")
        or original_input.get("input")
        or ""
    )


def _retrieve_documents(
    vector_store: Chroma,
    input_data: dict[str, Any],
    original_input: dict[str, Any],
) -> tuple[list[Document], str | None]:
    if input_data.get("skip_rag", False):
        skip_rag_manager.set_skip_rag(True)
        return [], None

    transaction = _transaction_text(input_data, original_input)
    if not transaction:
        skip_rag_manager.set_skip_rag(True)
        return [], "No transaction text was available for retrieval"

    # A non-L2 index is a configuration error, not an absence of similar cases.
    require_l2_collection(vector_store)
    try:
        candidates = vector_store.similarity_search_with_score(
            transaction,
            k=int(os.getenv("TXSHIELD_RETRIEVAL_K", "5")),
        )
    except Exception as exc:
        skip_rag_manager.set_skip_rag(True)
        return [], f"Retrieval failed ({type(exc).__name__})"

    distance_cutoff = float(CONFIG["L2_DISTANCE_THRESHOLD"])
    documents: list[Document] = []
    for document, score in candidates:
        squared_distance = float(score)
        if not math.isfinite(squared_distance) or squared_distance < 0.0:
            raise ValueError("Chroma returned an invalid squared L2 distance")
        distance = math.sqrt(squared_distance)
        # Chroma stores float32 vectors, so allow rounding at the inclusive
        # boundary (e.g. a true distance of 0.15 may return 0.150000003).
        if distance > distance_cutoff and not math.isclose(
            distance, distance_cutoff, rel_tol=1e-7, abs_tol=1e-9
        ):
            continue
        document.metadata = dict(document.metadata or {})
        document.metadata.update(
            similarity_score=distance,
            score=distance,
            chroma_squared_l2_distance=squared_distance,
            paper_similarity=1.0 - distance,
        )
        documents.append(document)

    skip_rag = not documents
    skip_rag_manager.set_skip_rag(skip_rag)
    input_data["skip_rag"] = skip_rag
    input_data["_retrieved_documents"] = documents
    return documents, None


def _assess_risk(
    transaction: str,
    documents: list[Document],
    xgb_reranker: XGBoostReranker | None,
) -> RiskGuidance:
    if xgb_reranker is None:
        return RiskGuidance()
    try:
        probability = float(xgb_reranker.predict_proba(transaction))
        threshold = float(
            getattr(
                xgb_reranker,
                "context_threshold",
                getattr(xgb_reranker, "threshold", 0.5),
            )
        )
        retained = probability >= threshold if documents else None
        return RiskGuidance(probability, threshold, retained)
    except Exception as exc:
        return RiskGuidance(error=f"XGBoost risk scoring failed ({type(exc).__name__})")


def create_conversational_chain(
    vector_store: Chroma,
    llm: Runnable,
    csv_file_path: str,
    original_input: dict[str, Any],
    xgb_reranker: XGBoostReranker | None = None,
) -> Runnable:
    """Create the pipeline with risk-guided context and LLM final decisions."""

    def run_pipeline(input_data: dict[str, Any]) -> dict[str, Any]:
        documents, retrieval_error = _retrieve_documents(
            vector_store,
            input_data,
            original_input,
        )
        transaction = _transaction_text(input_data, original_input)
        # Keep a computed risk hint even if its context is later omitted.
        guidance = (
            _assess_risk(transaction, documents, xgb_reranker)
            if documents else RiskGuidance()
        )

        context_documents = (
            [] if guidance.context_retained is False else documents
        )
        if guidance.context_retained is False:
            skip_rag_manager.set_skip_rag(True)

        reference_text = ""
        if context_documents:
            try:
                reference_text = _build_reference_text(
                    context_documents,
                    _load_descriptions(csv_file_path),
                )
            except (OSError, ValueError, KeyError) as exc:
                retrieval_error = f"Knowledge metadata failed ({type(exc).__name__})"
                context_documents = []
                skip_rag_manager.set_skip_rag(True)

        used_rag = bool(context_documents and reference_text and not retrieval_error)
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
            "xgb_prob": guidance.probability,
            "xgb_context_threshold": guidance.threshold,
        }
        llm_response = llm.invoke(llm_input)
        result: dict[str, Any] = {
            "answer": (
                llm_response.content
                if hasattr(llm_response, "content")
                else str(llm_response)
            ),
            "context": context_documents,
            "used_rag": used_rag,
            "similarity_info": _similarity_info(documents),
        }
        if retrieval_error:
            result["retrieval_error"] = retrieval_error
        if guidance.error:
            result["xgb_error"] = guidance.error
        if "_retrieved_documents" in input_data:
            result["_retrieved_documents"] = input_data["_retrieved_documents"]

        return apply_llm_decision(
            result,
            xgb_probability=guidance.probability,
            context_threshold=guidance.threshold,
            context_retained=guidance.context_retained,
        )

    return RunnableLambda(run_pipeline)
