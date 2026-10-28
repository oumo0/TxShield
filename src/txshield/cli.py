import argparse
from pathlib import Path
from typing import Any

from txshield.common.logging_config import configure_logging
from txshield.config.yaml_loader import cli_or_config, load_yaml_mapping
from txshield.detection.training.trainer import train
from txshield.preprocessing.pipeline import process_directory


def _load_config(explicit_path: str | None, default_path: str) -> dict[str, Any]:
    path = Path(explicit_path or default_path)
    if not explicit_path and not path.is_file():
        return {}
    return load_yaml_mapping(path)


def _mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"The YAML '{key}' value must be a mapping")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="txshield")
    sub = parser.add_subparsers(dest="command", required=True)

    preprocess = sub.add_parser("preprocess")
    preprocess.add_argument("--config")
    preprocess.add_argument("--input")
    preprocess.add_argument("--output")
    preprocess.add_argument("--target-function")

    training = sub.add_parser("train")
    training.add_argument("--config")
    training.add_argument("--train")
    training.add_argument("--test")
    training.add_argument("--model")
    training.add_argument("--threshold")
    training.add_argument("--reports")
    training.add_argument("--random-seed", type=int)
    training.add_argument("--validation-size", type=float)
    training.add_argument("--cv-folds", type=int)
    training.add_argument("--scoring")
    training.add_argument("--training-mode", choices=("paper", "grid_search"))
    return parser


def _run_preprocessing(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config, "configs/preprocessing.yaml")
    paths = _mapping(config, "paths")
    input_dir = args.input or paths.get("input")
    output_dir = args.output or paths.get("output")
    if not input_dir or not output_dir:
        raise ValueError("input and output must be set in YAML or on the command line")
    pruning_config = {
        key: value
        for key, value in config.items()
        if key not in {"paths", "target_function"}
    }
    return process_directory(
        input_dir,
        output_dir,
        cli_or_config(
            args.target_function,
            config,
            "target_function",
            "executeOperation",
        ),
        pruning_config,
    )


def _run_training(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_config(args.config, "configs/training.yaml")
    paths = _mapping(config, "paths")
    model_config = _mapping(config, "model")
    train_dir = args.train or paths.get("train")
    test_dir = args.test or paths.get("test")
    if not train_dir or not test_dir:
        raise ValueError("train and test must be set in YAML or on the command line")
    return train(
        train_dir,
        test_dir,
        args.model or paths.get("model", "artifacts/models/xgb_final_model.pkl"),
        args.threshold
        or paths.get("threshold", "artifacts/thresholds/best_threshold.txt"),
        args.reports or paths.get("report_dir", "artifacts/reports"),
        random_seed=cli_or_config(args.random_seed, config, "random_seed", 42),
        validation_size=cli_or_config(
            args.validation_size,
            config,
            "validation_size",
            0.2,
        ),
        cv_folds=cli_or_config(args.cv_folds, config, "cv_folds", 5),
        scoring=cli_or_config(args.scoring, config, "scoring", "f1"),
        model_params=model_config,
        training_mode=cli_or_config(
            args.training_mode,
            config,
            "training_mode",
            "paper",
        ),
    )


def main() -> None:
    args = _build_parser().parse_args()
    configure_logging()
    try:
        result = (
            _run_preprocessing(args)
            if args.command == "preprocess"
            else _run_training(args)
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(result)
    if result.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
