from pathlib import Path
import json
from typing import Any

def read_json(path: str | Path) -> Any:
    path = Path(path)
    # Decode UTF-8 strictly first; silently ignoring bytes can corrupt JSON content.
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="latin-1"))

def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
