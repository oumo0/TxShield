import argparse
import numpy as np
from txshield.detection.data.dataset_loader import load_data_by_filename
from txshield.detection.features.transformer import extract_and_transform
from txshield.detection.models.model_io import load_model, load_threshold
from txshield.detection.evaluation.metrics import evaluate_predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", default=None)
    parser.add_argument("--test", required=True)
    args = parser.parse_args()
    texts, labels = load_data_by_filename(args.test)
    model = load_model(args.model)
    threshold = load_threshold(args.threshold) if args.threshold else 0.5
    probabilities = model.predict_proba(extract_and_transform(texts))[:, 1]
    print(evaluate_predictions(np.asarray(labels), probabilities, threshold))

if __name__ == "__main__":
    main()
