import pytest

from txshield.detection.training.configuration import (
    PAPER_XGB_PARAMS,
    resolve_paper_params,
    resolve_training_mode,
)


def test_paper_parameters_match_reported_configuration():
    assert PAPER_XGB_PARAMS == {
        "max_depth": 3,
        "learning_rate": 0.02,
        "min_child_weight": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "reg_alpha": 0,
        "reg_lambda": 0.5,
    }


def test_paper_mode_allows_only_non_conflicting_overrides():
    params = resolve_paper_params({"n_estimators": 500, "max_depth": 3})
    assert params["n_estimators"] == 500
    assert params["max_depth"] == 3


def test_paper_mode_rejects_parameter_drift():
    with pytest.raises(ValueError, match="do not match the paper"):
        resolve_paper_params({"subsample": 0.8})


@pytest.mark.parametrize("value", ["paper", "grid_search"])
def test_supported_training_modes(value):
    assert resolve_training_mode(value) == value


def test_unknown_training_mode_is_rejected():
    with pytest.raises(ValueError, match="training_mode"):
        resolve_training_mode("automatic")
