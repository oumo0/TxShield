from typing import Any
from txshield.common.file_utils import read_json, write_json

def _signature(node: dict) -> tuple:
    return tuple(node.get(key, "") for key in
                 ("type", "contract", "function", "args", "value", "return_value"))

def mark_duplicate_calls(node: Any, seen: set[tuple]) -> int:
    if not isinstance(node, dict):
        return 0
    signature = _signature(node)
    marked = 0
    if signature in seen:
        if not node.get("_duplicate"):
            node["_duplicate"] = True
            marked += 1
    else:
        seen.add(signature)
        node.pop("_duplicate", None)
    children = node.get("internal_calls", [])
    if isinstance(children, list):
        for child in children:
            marked += mark_duplicate_calls(child, seen)
    return marked

def clean_call_log_data(data: dict) -> tuple[dict, int]:
    if isinstance(data.get("logs"), dict) and isinstance(data["logs"].get("calls"), list):
        roots = data["logs"]["calls"]
    elif isinstance(data.get("calls"), list):
        roots = data["calls"]
    elif isinstance(data.get("function_call"), list):
        roots = data["function_call"]
    else:
        return data, 0
    total = 0
    # Reset duplicate scope for each top-level transaction call, matching original behavior.
    for root in roots:
        total += mark_duplicate_calls(root, set())
    return data, total

def clean_call_log_file(source: str, output: str) -> int:
    data, count = clean_call_log_data(read_json(source))
    write_json(output, data)
    return count
