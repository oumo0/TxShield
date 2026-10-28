import json
from typing import Any

from txshield.common.file_utils import read_json, write_json
from txshield.common.trace_utils import child_call_key


def _signature(node: dict[str, Any]) -> tuple[Any, ...]:
    """Hash the six fields with stable object keys and ordered array values."""
    fields = (
        node.get("type", ""),
        node.get("contract", ""),
        node.get("function", node.get("func", "")),
        node.get("args", ""),
        node.get("value", ""),
        node.get("return", node.get("return_value", "")),
    )
    return tuple(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for value in fields
    )


def mark_duplicate_calls(node: Any, seen: set[tuple[Any, ...]]) -> int:
    if not isinstance(node, dict):
        return 0
    marked = 0
    stack = [node]
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        signature = _signature(current)
        if signature in seen:
            if not current.get("_duplicate"):
                current["_duplicate"] = True
                marked += 1
            # Algorithm 1 removes the duplicate node together with its subtree;
            # descendants must not affect duplicate detection in later branches.
            continue
        seen.add(signature)
        current.pop("_duplicate", None)
        children = current.get(child_call_key(current), [])
        if isinstance(children, list):
            stack.extend(reversed(children))
    return marked


def clean_call_log_data(data: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Mark duplicates independently within each top-level call tree."""
    if isinstance(data.get("logs"), dict) and isinstance(data["logs"].get("calls"), list):
        roots = data["logs"]["calls"]
    elif isinstance(data.get("calls"), list):
        roots = data["calls"]
    elif isinstance(data.get("function_call"), list):
        roots = data["function_call"]
    else:
        return data, 0
    total = 0
    for root in roots:
        total += mark_duplicate_calls(root, set())
    return data, total


def clean_call_log_file(source: str, output: str) -> int:
    data, count = clean_call_log_data(read_json(source))
    write_json(output, data)
    return count
