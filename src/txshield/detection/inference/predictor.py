from pathlib import Path

from txshield.common.trace_utils import validate_trace_text
from txshield.detection.features.extractor import extract_features_from_json
from txshield.detection.features.transformer import transform_feature_row
from txshield.detection.models.model_io import load_model, load_threshold

class AttackPredictor:
    """Standalone XGBoost predictor for numerical baseline evaluation."""

    def __init__(
        self,
        model_path: str | Path,
        threshold_path: str | Path | None = None,
        default_threshold: float = 0.5,
    ) -> None:
        self.model = load_model(model_path)
        self.threshold = (
            load_threshold(threshold_path, default_threshold)
            if threshold_path
            else float(default_threshold)
        )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1")

    def predict_proba(self, json_text: str) -> float:
        validate_trace_text(json_text)
        row = transform_feature_row(extract_features_from_json(json_text)).reshape(
            1,
            -1,
        )
        return float(self.model.predict_proba(row)[0, 1])

    def predict(self, json_text: str) -> dict[str, float | bool]:
        probability = self.predict_proba(json_text)
        return {
            "attack_probability": probability,
            "threshold": self.threshold,
            "is_attack": probability >= self.threshold,
        }
