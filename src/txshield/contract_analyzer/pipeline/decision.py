from __future__ import annotations

import re
from dataclasses import dataclass

_LABEL = re.compile(r"^\s*(?:\[\s*)?(Attack|Benign)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Decision:
    """The LLM label together with the pre-inference XGBoost evidence."""
    llm_prediction: str
    final_prediction: str
    xgb_attack_probability: float | None
    xgb_threshold: float | None
    xgb_applied: bool
    xgb_decision: str


def parse_prediction(answer: str) -> str:
    """Read the required leading label; do not infer labels from prose."""
    match = _LABEL.match(answer or "")
    return match.group(1).capitalize() if match else "Unknown"


def record_llm_decision(
    answer: str,
    xgb_probability: float | None,
    context_threshold: float | None,
    context_retained: bool | None,
) -> Decision:
    """Record the LLM's final label without allowing XGBoost to overwrite it."""
    label = parse_prediction(answer)
    if xgb_probability is None:
        return Decision(label, label, None, None, False, "unavailable")
    if context_retained is None:
        context_decision = "no_eligible_context"
    else:
        context_decision = "retain_context" if context_retained else "omit_context"
    return Decision(
        label,
        label,
        float(xgb_probability),
        float(context_threshold) if context_threshold is not None else None,
        True,
        context_decision,
    )


def apply_llm_decision(
    result: dict,
    *,
    xgb_probability: float | None,
    context_threshold: float | None,
    context_retained: bool | None,
) -> dict:
    """Preserve the LLM answer and expose XGBoost guidance metadata."""
    decision = record_llm_decision(
        str(result.get("answer", "")),
        xgb_probability,
        context_threshold,
        context_retained,
    )
    result.update({
        "llm_prediction": decision.llm_prediction,
        "final_prediction": decision.final_prediction,
        "xgb_applied": decision.xgb_applied,
        "xgb_decision": decision.xgb_decision,
        "xgb_attack_probability": decision.xgb_attack_probability,
        "xgb_threshold": decision.xgb_threshold,
        "xgb_prob": decision.xgb_attack_probability,
        "xgb_context_retained": context_retained,
    })
    return result
