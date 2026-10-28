import json
from typing import Any


def child_call_key(node: dict[str, Any]) -> str:
    """Accept either child schema without silently dropping a second tree."""
    if node.get("internal_calls") and node.get("calls"):
        raise ValueError("Use either internal_calls or calls for a node's children")
    if node.get("internal_calls") or "calls" not in node:
        return "internal_calls"
    return "calls"


def validate_trace_text(text: str) -> dict[str, Any]:
    """Require a nonempty call tree before model inference or training."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Transaction trace is empty")
    try:
        data = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise ValueError("Transaction trace must be valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("Transaction trace must be a JSON object")
    logs = data.get("logs")
    candidates = [data.get("function_call"), data.get("calls")]
    if isinstance(logs, dict):
        candidates.append(logs.get("calls"))
    if any(value is not None and not isinstance(value, list) for value in candidates):
        raise ValueError("Trace roots must be a list of call objects")
    roots = [value for value in candidates if value]
    if len(roots) != 1:
        raise ValueError("Provide one nonempty function_call, calls, or logs.calls list")
    stack = list(roots[0])
    while stack:
        node = stack.pop()
        if not isinstance(node, dict) or not node:
            raise ValueError("Each call must be a nonempty JSON object")
        children = node.get(child_call_key(node), [])
        if not isinstance(children, list):
            raise ValueError("Call children must be a list")
        stack.extend(children)
    return data
