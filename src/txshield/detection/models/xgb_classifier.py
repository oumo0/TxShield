from xgboost import XGBClassifier

def build_classifier(random_seed: int = 42, **overrides) -> XGBClassifier:
    params = {
        "random_state": random_seed,
        "eval_metric": "logloss",
        "n_estimators": 500,
        "n_jobs": -1,
        "verbosity": 0,
    }
    params.update(overrides)
    return XGBClassifier(**params)
