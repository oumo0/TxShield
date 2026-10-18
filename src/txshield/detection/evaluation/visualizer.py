from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def save_roc_curve(y_true, probabilities, output_path: str | Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    score = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path); plt.close()

def save_precision_recall_curve(y_true, probabilities, output_path: str | Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path); plt.close()
