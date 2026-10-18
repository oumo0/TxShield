"""Validated XGBoost configurations for paper reproduction and exploration."""
from __future__ import annotations

from typing import Any


PAPER_XGB_PARAMS: dict[str, Any] = {
    "max_depth": 3,
    "learning_rate": 0.02,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.7,
    "reg_alpha": 0,
    "reg_lambda": 0.5,
}


def resolve_training_mode(value: str) -> str:
    """Normalize and validate the supported training modes."""
    mode = str(value).strip().lower()
    if mode not in {"paper", "grid_search"}:
        raise ValueError("training_mode must be 'paper' or 'grid_search'")
    return mode


def resolve_paper_params(configured: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return paper parameters and reject conflicting reproduction settings.

    Additional implementation parameters such as ``n_estimators`` and
    ``n_jobs`` are allowed, but the seven parameters reported by the paper
    cannot be changed in paper mode.
    """
    params = dict(configured or {})
    conflicts = {
        name: (params[name], expected)
        for name, expected in PAPER_XGB_PARAMS.items()
        if name in params and params[name] != expected
    }
    if conflicts:
        details = ", ".join(
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in conflicts.items()
        )
        raise ValueError(f"Paper-mode XGBoost parameters do not match the paper: {details}")
    return {**params, **PAPER_XGB_PARAMS}
