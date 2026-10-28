from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def _prepare_output(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def save_roc_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_path: str | Path,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    score = auc(fpr, tpr)
    figure, axis = plt.subplots()
    axis.plot(fpr, tpr, label=f"AUC = {score:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--")
    axis.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(_prepare_output(output_path))
    plt.close(figure)


def save_precision_recall_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_path: str | Path,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    figure, axis = plt.subplots()
    axis.plot(recall, precision)
    axis.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    figure.tight_layout()
    figure.savefig(_prepare_output(output_path))
    plt.close(figure)
