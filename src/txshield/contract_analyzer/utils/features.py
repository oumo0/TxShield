"""Compatibility wrapper around the canonical 14-feature extractor."""
from txshield.detection.features.extractor import extract_features_from_json
from txshield.detection.features.signatures import SIG_MAP, safe_parse_hex


def extract_features_from_json_v2(json_str: str) -> list[float]:
    """Preserve the original 321 function name for existing callers."""
    return extract_features_from_json(json_str)


__all__ = ["SIG_MAP", "safe_parse_hex", "extract_features_from_json_v2"]
