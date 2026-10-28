import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    """Read JSON as UTF-8, with a legacy Latin-1 fallback."""
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="latin-1"))


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
