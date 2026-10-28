from pathlib import Path
import pickle

from xgboost import XGBClassifier


def save_model(model: XGBClassifier, path: str | Path) -> None:
    """Save pickle artifacts as pickle and JSON/UBJ with XGBoost's writer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("wb") as output:
            pickle.dump(model, output, protocol=pickle.HIGHEST_PROTOCOL)
        return
    model.save_model(path)


def load_model(path: str | Path) -> XGBClassifier:
    """Load trusted pickle artifacts or XGBoost native model files."""
    path = Path(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as source:
            return pickle.load(source)
    model = XGBClassifier()
    model.load_model(path)
    return model


def save_threshold(threshold: float, path: str | Path) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{threshold:.8f}\n", encoding="utf-8")


def load_threshold(path: str | Path, default: float = 0.5) -> float:
    path = Path(path)
    threshold = (
        float(path.read_text(encoding="utf-8").strip())
        if path.exists()
        else float(default)
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1")
    return threshold
