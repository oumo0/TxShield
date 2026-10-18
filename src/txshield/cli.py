import argparse
from txshield.common.logging_config import configure_logging
from txshield.preprocessing.pipeline import process_directory
from txshield.detection.training.trainer import train

def main():
    parser = argparse.ArgumentParser(prog="txshield")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("preprocess")
    prep.add_argument("--input", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--target-function", default="executeOperation")
    tr = sub.add_parser("train")
    tr.add_argument("--train", required=True)
    tr.add_argument("--test", required=True)
    tr.add_argument("--model", default="artifacts/models/xgb_model.json")
    tr.add_argument("--threshold", default="artifacts/thresholds/best_threshold.txt")
    tr.add_argument("--reports", default="artifacts/reports")
    args = parser.parse_args()
    configure_logging()
    if args.command == "preprocess":
        print(process_directory(args.input, args.output, args.target_function))
    elif args.command == "train":
        print(train(args.train, args.test, args.model, args.threshold, args.reports))

if __name__ == "__main__":
    main()
