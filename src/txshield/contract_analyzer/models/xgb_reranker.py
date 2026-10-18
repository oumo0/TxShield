"""XGBoost post-filter used after an LLM predicts an attack."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

from txshield.contract_analyzer.utils.features import extract_features_from_json_v2
from txshield.detection.features.extractor import FEATURE_NAMES
from txshield.detection.features.transformer import (
    DROP_COLUMNS,
    MODEL_FEATURE_NAMES as CANONICAL_MODEL_FEATURE_NAMES,
    transform_feature_row,
)


_XGB_RERANKER = None

RAW_FEATURE_NAMES = tuple(FEATURE_NAMES)
MODEL_DROP_INDICES = tuple(DROP_COLUMNS)
MODEL_FEATURE_NAMES = tuple(CANONICAL_MODEL_FEATURE_NAMES)


def _load_model(path: Path):
    """Load legacy pickle models or XGBoost JSON/UBJ models."""
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as source:
            return pickle.load(source)
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(path)
    return model

class XGBoostReranker:
    """Score LLM attack candidates with the trained XGBoost model."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        threshold_path: str | os.PathLike[str] | None = None,
    ) -> None:
        resolved_model_path = Path(
            model_path
            or os.getenv("TXSHIELD_XGB_MODEL", "artifacts/models/xgb_final_model.pkl")
        )
        self.model = _load_model(resolved_model_path)

        configured_threshold = os.getenv("TXSHIELD_XGB_THRESHOLD")
        if configured_threshold is not None:
            self.threshold = float(configured_threshold)
        else:
            configured_path = threshold_path or os.getenv("TXSHIELD_XGB_THRESHOLD_PATH")
            candidates = (
                [Path(configured_path)]
                if configured_path
                else [
                    Path("best_threshold.txt"),
                    Path("artifacts/thresholds/best_threshold.txt"),
                    resolved_model_path.with_name("best_threshold.txt"),
                ]
            )
            selected_path = next((path for path in candidates if path.is_file()), None)
            self.threshold = (
                float(selected_path.read_text(encoding="utf-8").strip())
                if selected_path is not None
                else 0.5
            )

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("XGBoost threshold must be between 0 and 1")
        self.raw_feature_names = RAW_FEATURE_NAMES
        self.feature_names = MODEL_FEATURE_NAMES

    def predict_proba(self, json_str: str) -> float:
        """Return the attack probability for one transaction JSON string."""
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

