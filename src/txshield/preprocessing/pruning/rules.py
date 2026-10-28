import re
from typing import Any

from txshield.common.trace_utils import child_call_key
from txshield.detection.features.signatures import safe_parse_hex
DEFAULT_CONFIG = {
    "remove_static_metadata": True,
    "max_depth": 12,
    "shorten_args": True,
    "max_arg_length": 200,
    "address_alias": True,
    "remove_duplicate_marked": True,
}

def apply_general_rules(node: Any, current_depth: int = 0,
                        config: dict | None = None) -> Any:
    config = {**DEFAULT_CONFIG, **(config or {})}
    if not isinstance(node, dict):
        return node
    if config["remove_duplicate_marked"] and node.get("_duplicate") is True:
        return None
    max_depth = config.get("max_depth")
    if max_depth is not None and current_depth > max_depth:
        return {"_truncated": f"depth {current_depth} exceeds limit"}

    func_name = safe_parse_hex(node.get("function", node.get("func", "")))
    func_name = func_name.split("(", 1)[0].strip().lower()
    if config["remove_static_metadata"] and node.get("type", "").upper() == "STATICCALL":
        if func_name in {"decimals", "symbol", "name"}:
            return None

    children_key = child_call_key(node)
    children = node.get(children_key, [])
    if isinstance(children, list):
        node[children_key] = [
            result for child in children
            if (result := apply_general_rules(child, current_depth + 1, config)) is not None
        ]

    args = node.get("args")
    if config["shorten_args"] and isinstance(args, str):
        limit = int(config["max_arg_length"])
        if len(args) > limit:
            node["args"] = args[:limit] + "...<truncated>"

    if config["address_alias"] and isinstance(node.get("contract"), str):
        address = node["contract"]
        match = re.search(r"0x[a-fA-F0-9]{8,}", address)
        if match and len(match.group(0)) > 10:
            raw = match.group(0)
            node["contract"] = address.replace(raw, raw[:6] + "..." + raw[-4:])
    return node
