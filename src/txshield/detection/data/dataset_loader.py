from pathlib import Path

from txshield.common.trace_utils import validate_trace_text

def load_data_by_filename(data_dir: str | Path) -> tuple[list[str], list[int]]:
    """Load serialized traces labeled by ``*_attack`` or ``*_benign`` suffix."""
    texts, labels = [], []
    for path in sorted(Path(data_dir).glob("*.json")):
        if path.name.endswith("_attack.json"):
            label = 1
        elif path.name.endswith("_benign.json"):
            label = 0
        else:
            continue
        text = path.read_text(encoding="utf-8")
        validate_trace_text(text)
        texts.append(text)
        labels.append(label)
    if not texts:
        raise ValueError(f"No *_attack.json or *_benign.json files found in {data_dir}")
    return texts, labels
