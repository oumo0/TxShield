import argparse
from txshield.common.logging_config import configure_logging
from txshield.config.yaml_loader import cli_or_config, load_yaml_mapping
from txshield.preprocessing.pipeline import process_directory

def main():
    parser = argparse.ArgumentParser(description="Preprocess transaction execution traces.")
    parser.add_argument("--config", default="configs/preprocessing.yaml")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--target-function")
    args = parser.parse_args()
    config = load_yaml_mapping(args.config)
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        parser.error("The YAML 'paths' value must be a mapping")
    input_dir = args.input or paths.get("input")
    output_dir = args.output or paths.get("output")
    if not input_dir or not output_dir:
        parser.error("input and output must be set in YAML or with --input/--output")
    target_function = cli_or_config(
        args.target_function, config, "target_function", "executeOperation"
    )
    pruning_config = {
        key: value
        for key, value in config.items()
        if key not in {"paths", "target_function"}
    }
    configure_logging()
    result = process_directory(input_dir, output_dir, target_function, pruning_config)
    print(result)
    if result["failed"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
