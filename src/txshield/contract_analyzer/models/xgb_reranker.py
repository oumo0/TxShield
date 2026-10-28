from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from txshield.common.trace_utils import validate_trace_text
from txshield.contract_analyzer.utils.features import extract_features_from_json_v2
from txshield.detection.features.extractor import FEATURE_NAMES
from txshield.detection.models.model_io import load_model
from txshield.detection.features.transformer import (
    DROP_COLUMNS,
    MODEL_FEATURE_NAMES as CANONICAL_MODEL_FEATURE_NAMES,
    transform_feature_row,
)


_XGB_RERANKER = None

RAW_FEATURE_NAMES = tuple(FEATURE_NAMES)
MODEL_DROP_INDICES = tuple(DROP_COLUMNS)
MODEL_FEATURE_NAMES = tuple(CANONICAL_MODEL_FEATURE_NAMES)


class XGBoostReranker:
    """Estimate transaction risk for context selection and LLM prompting."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        context_threshold: float | None = None,
    ) -> None:
        resolved_model_path = Path(
            model_path
            or os.getenv("TXSHIELD_XGB_MODEL", "artifacts/models/xgb_final_model.pkl")
        )
        self.model = load_model(resolved_model_path)

        configured_threshold = context_threshold
        if configured_threshold is None:
            configured_threshold = float(
                os.getenv(
                    "TXSHIELD_XGB_CONTEXT_THRESHOLD",
                    os.getenv("TXSHIELD_XGB_THRESHOLD", "0.5"),
                )
            )
        self.context_threshold = float(configured_threshold)
        self.threshold = self.context_threshold
        if not 0.0 <= self.context_threshold <= 1.0:
            raise ValueError("XGBoost context threshold must be between 0 and 1")
        self.raw_feature_names = RAW_FEATURE_NAMES
        self.feature_names = MODEL_FEATURE_NAMES

    def predict_proba(self, json_str: str) -> float:
        """Return the attack probability for one transaction JSON string."""
        validate_trace_text(json_str)
        raw_features = np.asarray(extract_features_from_json_v2(json_str), dtype=float)
        if raw_features.shape != (len(RAW_FEATURE_NAMES),):
            raise ValueError(
                f"Expected {len(RAW_FEATURE_NAMES)} raw features, got {raw_features.size}"
            )
        model_features = transform_feature_row(raw_features.tolist()).reshape(1, -1)
        return float(self.model.predict_proba(model_features)[0, 1])

def get_xgb_reranker() -> XGBoostReranker:
    global _XGB_RERANKER
    if _XGB_RERANKER is None:
        _XGB_RERANKER = XGBoostReranker()
    return _XGB_RERANKER

