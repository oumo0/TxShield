import argparse
from typing import Any

from txshield.config.yaml_loader import cli_or_config, load_yaml_mapping

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the XGBoost classifier.")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--train")
    parser.add_argument("--test")
    parser.add_argument("--model")
    parser.add_argument("--threshold")
    parser.add_argument("--reports")
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--validation-size", type=float)
    parser.add_argument("--cv-folds", type=int)
    parser.add_argument("--scoring")
    parser.add_argument("--training-mode", choices=("paper", "grid_search"))
    args = parser.parse_args()
    config: dict[str, Any] = load_yaml_mapping(args.config)
    paths = config.get("paths", {})
    model_config = config.get("model", {})
    if not isinstance(paths, dict) or not isinstance(model_config, dict):
        parser.error("The YAML 'paths' and 'model' values must be mappings")
    train_dir = args.train or paths.get("train")
    test_dir = args.test or paths.get("test")
    if not train_dir or not test_dir:
        parser.error("train and test must be set in YAML or with --train/--test")
    from txshield.detection.training.trainer import train

    print(train(
        train_dir,
        test_dir,
        args.model or paths.get("model", "artifacts/models/xgb_model.json"),
        args.threshold or paths.get("threshold", "artifacts/thresholds/best_threshold.txt"),
        args.reports or paths.get("report_dir", "artifacts/reports"),
        random_seed=cli_or_config(args.random_seed, config, "random_seed", 42),
        validation_size=cli_or_config(args.validation_size, config, "validation_size", 0.2),
        cv_folds=cli_or_config(args.cv_folds, config, "cv_folds", 5),
        scoring=cli_or_config(args.scoring, config, "scoring", "f1"),
        model_params=model_config,
        training_mode=cli_or_config(
            args.training_mode, config, "training_mode", "paper"
        ),
    ))

if __name__ == "__main__":
    main()
