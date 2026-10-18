from sklearn.model_selection import GridSearchCV
from txshield.detection.models.xgb_classifier import build_classifier

def run_grid_search(X, y, random_seed: int = 42, cv_folds: int = 5,
                    scoring: str = "f1", model_params: dict | None = None):
    # No fixed eval_set is passed to GridSearchCV: doing so would leak the same
    # holdout into every fold's early stopping decision.
    base_params = dict(model_params or {})
    base_params.pop("random_state", None)
    model = build_classifier(random_seed=random_seed, **base_params)
    grid = {
        "max_depth": [3, 4],
        "learning_rate": [0.005, 0.01, 0.02, 0.05],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8],
        "reg_alpha": [0, 0.5, 1.0, 2.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
        "min_child_weight": [1, 3, 5, 7],
    }
    search = GridSearchCV(model, grid, scoring=scoring, cv=cv_folds, n_jobs=-1, verbose=1)
    search.fit(X, y)
    return search
