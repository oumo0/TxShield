import logging
from pathlib import Path
from typing import Any

from txshield.common.file_utils import read_json, write_json
from txshield.preprocessing.cleaning.call_cleaner import clean_call_log_data
from txshield.preprocessing.pruning.call_chain import prune_data

logger = logging.getLogger(__name__)


def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    target_function: str = "executeOperation",
    pruning_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Process JSON files into a separate output directory; raw inputs are never edited."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError("Input directory does not exist")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Input and output directories must be different")
    sources = sorted(input_dir.glob("*.json"))
    if not sources:
        raise ValueError("Input directory contains no JSON files")
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"processed": 0, "failed": 0, "errors": []}
    for source in sources:
        try:
            data = read_json(source)
            data, marked = clean_call_log_data(data)
            data = prune_data(data, target_function, pruning_config)
            write_json(output_dir / source.name, data)
            stats["processed"] += 1
            logger.info(
                "Processed %s (duplicate nodes marked: %d)",
                source.name,
                marked,
            )
        except Exception as exc:
            stats["failed"] += 1
            stats["errors"].append({"file": source.name, "error": str(exc)})
            logger.exception("Failed to process %s", source.name)
    return stats
