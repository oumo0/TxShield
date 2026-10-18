import numpy as np
from txshield.detection.features.extractor import extract_features_from_json
from txshield.detection.features.transformer import transform_feature_row
from txshield.detection.models.model_io import load_model, load_threshold

class AttackPredictor:
    def __init__(self, model_path, threshold_path=None, default_threshold=0.5):
        self.model = load_model(model_path)
        self.threshold = load_threshold(threshold_path, default_threshold) if threshold_path else default_threshold

    def predict_proba(self, json_text: str) -> float:
        row = transform_feature_row(extract_features_from_json(json_text)).reshape(1, -1)
        return float(self.model.predict_proba(row)[0, 1])

    def predict(self, json_text: str) -> dict:
        probability = self.predict_proba(json_text)
        return {
            "attack_probability": probability,
            "threshold": self.threshold,
            "is_attack": probability >= self.threshold,
        }
