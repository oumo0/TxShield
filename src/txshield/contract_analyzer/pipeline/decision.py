"""Paper-aligned post-LLM XGBoost attack filtering."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_LABEL = re.compile(r"^\s*(?:\[\s*)?(Attack|Benign)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Decision:
    llm_prediction: str
    final_prediction: str
    xgb_attack_probability: float | None
    xgb_threshold: float | None
    xgb_applied: bool
    xgb_decision: str


def parse_prediction(answer: str) -> str:
    """Read the required leading label; do not infer labels from explanation text."""
    match = _LABEL.match(answer or "")
    return match.group(1).capitalize() if match else "Unknown"


def post_filter(answer: str, transaction_json: str, reranker: Any | None) -> Decision:
    """Only filter LLM attack candidates; benign/unknown results bypass XGBoost."""
    label = parse_prediction(answer)
    if label != "Attack" or reranker is None:
        return Decision(label, label, None, None, False, "not_applied")
    probability = float(reranker.predict_proba(transaction_json))
    threshold = float(reranker.threshold)
    accepted = probability >= threshold
    return Decision(label, "Attack" if accepted else "Benign", probability,
                    threshold, True, "accept_attack" if accepted else "reject_attack")


def apply_post_filter(result: dict, transaction_json: str, reranker: Any | None) -> dict:
    """Preserve the original LLM answer while adding an explicit final decision."""
    decision = post_filter(str(result.get("answer", "")), transaction_json, reranker)
    result.update({
        "llm_prediction": decision.llm_prediction,
        "final_prediction": decision.final_prediction,
        "xgb_applied": decision.xgb_applied,
        "xgb_decision": decision.xgb_decision,
        "xgb_attack_probability": decision.xgb_attack_probability,
        "xgb_threshold": decision.xgb_threshold,
        "xgb_prob": decision.xgb_attack_probability,
    })
    return result
