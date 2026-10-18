from pathlib import Path
import json
from xgboost import XGBClassifier

def save_model(model: XGBClassifier, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)

def load_model(path: str | Path) -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(path)
    return model

def save_threshold(threshold: float, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{threshold:.8f}\\n", encoding="utf-8")

def load_threshold(path: str | Path, default: float = 0.5) -> float:
    path = Path(path)
    return float(path.read_text(encoding="utf-8").strip()) if path.exists() else default
