from pathlib import Path
import logging
from txshield.common.file_utils import read_json, write_json
from txshield.preprocessing.cleaning.call_cleaner import clean_call_log_data
from txshield.preprocessing.pruning.call_chain import prune_data

logger = logging.getLogger(__name__)

def process_directory(input_dir: str | Path, output_dir: str | Path,
                      target_function: str = "executeOperation",
                      pruning_config: dict | None = None) -> dict:
    """Process JSON files into a separate output directory; raw inputs are never edited."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"processed": 0, "failed": 0, "errors": []}
    for source in sorted(input_dir.glob("*.json")):
        try:
            data = read_json(source)
            data, marked = clean_call_log_data(data)
            data = prune_data(data, target_function, pruning_config)
            write_json(output_dir / source.name, data)
            stats["processed"] += 1
            logger.info("Processed %s (duplicate nodes marked: %d)", source.name, marked)
        except Exception as exc:
            stats["failed"] += 1
            stats["errors"].append({"file": source.name, "error": str(exc)})
            logger.exception("Failed to process %s", source.name)
    return stats
