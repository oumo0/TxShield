import numpy as np
from txshield.detection.features.extractor import extract_features_from_json, FEATURE_NAMES

DROP_COLUMNS = [7, 8, 9, 11]
MODEL_FEATURE_NAMES = [name for i, name in enumerate(FEATURE_NAMES) if i not in DROP_COLUMNS]

def transform_feature_row(features: list[float]) -> np.ndarray:
    values = np.asarray(features, dtype=float).copy()
    if values.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} features, got shape {values.shape}")
    values[10] = np.log1p(max(values[10], 0.0))
    values[11] = np.log1p(max(values[11], 0.0))
    values = np.delete(values, DROP_COLUMNS)
    values[~np.isfinite(values)] = 0.0
    return values

def extract_and_transform(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, len(MODEL_FEATURE_NAMES)), dtype=float)
    return np.vstack([transform_feature_row(extract_features_from_json(text)) for text in texts])
