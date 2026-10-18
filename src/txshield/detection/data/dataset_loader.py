from pathlib import Path
from txshield.common.file_utils import read_json

def load_data_by_filename(data_dir: str | Path) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    for path in sorted(Path(data_dir).glob("*.json")):
        if path.name.endswith("_attack.json"):
            label = 1
        elif path.name.endswith("_benign.json"):
            label = 0
        else:
            continue
        # Preserve original JSON text because the feature extractor expects serialized JSON.
        texts.append(path.read_text(encoding="utf-8", errors="replace"))
        labels.append(label)
    if not texts:
        raise ValueError(f"No *_attack.json or *_benign.json files found in {data_dir}")
    return texts, labels
