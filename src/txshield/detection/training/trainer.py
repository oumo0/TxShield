from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from txshield.detection.data.dataset_loader import load_data_by_filename
from txshield.detection.features.transformer import extract_and_transform
from txshield.detection.training.hyperparameter_search import run_grid_search
from txshield.detection.training.threshold_optimizer import find_best_f1_threshold
from txshield.detection.models.model_io import save_model, save_threshold
from txshield.detection.models.xgb_classifier import build_classifier
from txshield.detection.training.configuration import (
    resolve_paper_params,
    resolve_training_mode,
)

def train(train_dir, test_dir, model_path, threshold_path, report_dir,
          random_seed=42, validation_size=0.2, cv_folds=5, scoring="f1",
          model_params=None, training_mode="paper"):
    mode = resolve_training_mode(training_mode)
    if not 0.0 < float(validation_size) < 1.0:
        raise ValueError("validation_size must be between 0 and 1")
    if mode == "grid_search" and int(cv_folds) < 2:
        raise ValueError("cv_folds must be at least 2")
    train_texts, train_labels = load_data_by_filename(train_dir)
    test_texts, test_labels = load_data_by_filename(test_dir)
    X = extract_and_transform(train_texts)
    y = np.asarray(train_labels)
    X_test = extract_and_transform(test_texts)
    y_test = np.asarray(test_labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_size, random_state=random_seed, stratify=y
    )
    if mode == "paper":
        selected_params = resolve_paper_params(model_params)
        model = build_classifier(random_seed=random_seed, **selected_params)
        model.fit(X_train, y_train)
        cv_f1 = None
    else:
        search = run_grid_search(
            X_train, y_train, random_seed, cv_folds, scoring, model_params
        )
        model = search.best_estimator_
        selected_params = {**dict(model_params or {}), **search.best_params_}
        cv_f1 = float(search.best_score_)
    val_prob = model.predict_proba(X_val)[:, 1]
    threshold, val_f1 = find_best_f1_threshold(y_val, val_prob)

    # Refit the selected configuration on all training samples, then evaluate
    # exactly once on the held-out test set.
    final_params = dict(selected_params)
    final_params.pop("random_state", None)
    final_model = build_classifier(random_seed=random_seed, **final_params)
    final_model.fit(X, y)
    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_prob)
    save_model(final_model, model_path)
    save_threshold(threshold, threshold_path)
    report = {
        "training_mode": mode,
        "selected_params": selected_params,
        "best_params": selected_params,
        "cv_f1": cv_f1,
        "validation_f1_at_selected_threshold": float(val_f1),
        "threshold": float(threshold),
        "test_auc": float(test_auc),
        "test_samples": int(len(y_test)),
    }
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
