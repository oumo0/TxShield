from __future__ import annotations

import json
from typing import Any

import numpy as np

from txshield.detection.features.signatures import safe_parse_hex


FEATURE_NAMES = [
    "call_count", "event_count", "max_depth", "unique_contracts",
    "has_transfer", "has_approve", "has_flashloan", "has_delegatecall",
    "has_selfdestruct", "has_high_value", "total_value", "max_value",
    "unique_signatures", "unknown_sig_count",
]


def _parse_value(value: Any) -> float | None:
    try:
        parsed = (
            float(int(value, 16))
            if isinstance(value, str) and value.startswith("0x")
            else float(value)
        )
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if np.isfinite(parsed) and parsed <= 1e30 else None


def _find_calls(data: dict[str, Any]) -> list[Any]:
    logs = data.get("logs")
    candidates = (
        data.get("function_call"),
        data.get("calls"),
        logs.get("calls") if isinstance(logs, dict) else None,
    )
    for calls in candidates:
        if isinstance(calls, list) and calls:
            return calls
    return next((value for value in data.values() if isinstance(value, list)), [])


def extract_features_from_json(json_text: str) -> list[float]:
    """Return the 14 candidate features in stable training-time order.

    Call depth is one-based. Iterative traversal keeps training and runtime
    inference safe for unusually deep execution traces.
    """
    try:
        decoded = json.loads(json_text)
    except (TypeError, json.JSONDecodeError):
        decoded = {}
    data = decoded if isinstance(decoded, dict) else {}

    call_count = event_count = max_depth = 0
    contracts: set[str] = set()
    signatures: list[str] = []
    values: list[float] = []
    total_value = 0.0
    flags: set[str] = set()

    stack = [(call, 1) for call in reversed(_find_calls(data))]
    while stack:
        call, depth = stack.pop()
        if not isinstance(call, dict):
            continue
        call_count += 1
        max_depth = max(max_depth, depth)
        raw_function = call.get("func", call.get("function", ""))
        function_name = safe_parse_hex(raw_function) if raw_function else ""
        signatures.append(function_name)
        lowered_name = function_name.lower()

        contract = call.get("contract", call.get("to", ""))
        if isinstance(contract, str) and contract:
            contracts.add(contract)

        call_type = str(call.get("type", "")).upper()
        event_count += call_type == "EVENT"
        if "transfer" in lowered_name:
            flags.add("has_transfer")
        if "approve" in lowered_name:
            flags.add("has_approve")
        if "flash" in lowered_name:
            flags.add("has_flashloan")
        if call_type == "DELEGATECALL" or lowered_name == "delegatecall":
            flags.add("has_delegatecall")
        if "selfdestruct" in lowered_name:
            flags.add("has_selfdestruct")

        parsed_value = _parse_value(call.get("value", "0"))
        if parsed_value is not None:
            values.append(parsed_value)
            total_value += parsed_value
            if parsed_value > 1e18:
                flags.add("has_high_value")

        children = call.get("internal_calls") or call.get("calls") or []
        if isinstance(children, list):
            stack.extend((child, depth + 1) for child in reversed(children))

    features = [
        call_count,
        event_count,
        max_depth,
        len(contracts),
        int("has_transfer" in flags),
        int("has_approve" in flags),
        int("has_flashloan" in flags),
        int("has_delegatecall" in flags),
        int("has_selfdestruct" in flags),
        int("has_high_value" in flags),
        min(total_value, 1e30),
        min(max(values, default=0.0), 1e30),
        len(set(signatures)),
        sum(
            isinstance(signature, str)
            and signature.startswith("0x")
            and len(signature) >= 10
            for signature in signatures
        ),
    ]
    return [0.0 if not np.isfinite(value) else float(value) for value in features]
