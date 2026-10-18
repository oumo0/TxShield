"""Regression tests for paper-aligned final decision behavior."""
import numpy as np
import pytest

from txshield.contract_analyzer.models.xgb_reranker import (
    MODEL_DROP_INDICES,
    MODEL_FEATURE_NAMES,
    RAW_FEATURE_NAMES,
    XGBoostReranker,
)
from txshield.contract_analyzer.pipeline.decision import post_filter, apply_post_filter
from txshield.contract_analyzer.utils.features import extract_features_from_json_v2


class Reranker:
    threshold = 0.5
    def __init__(self, probability):
        self.probability = probability
        self.calls = 0
    def predict_proba(self, value):
        self.calls += 1
        return self.probability


def test_attack_rejected_by_xgb():
    model = Reranker(0.2)
    result = apply_post_filter({"answer": "Attack\nSuspicious transfer", "used_rag": True}, "{}", model)
    assert result["llm_prediction"] == "Attack"
    assert result["final_prediction"] == "Benign"
    assert result["answer"].startswith("Attack")
    assert result["used_rag"] is True
    assert result["xgb_decision"] == "reject_attack"
    assert model.calls == 1


def test_attack_accepted_by_xgb():
    assert post_filter("[Attack]", "{}", Reranker(0.8)).final_prediction == "Attack"


def test_benign_does_not_invoke_xgb():
    model = Reranker(0.99)
    result = post_filter("Benign\nOrdinary swap", "{}", model)
    assert result.final_prediction == "Benign"
    assert result.xgb_applied is False
    assert model.calls == 0


def test_unparseable_response_not_classified():
    assert post_filter("Not enough evidence", "{}", Reranker(0.9)).final_prediction == "Unknown"


def test_successful_analysis_result_is_returned(monkeypatch):
    pytest.importorskip("langchain_core")
    from txshield.contract_analyzer import main

    expected = {
        "answer": "Benign\nOrdinary protocol interaction",
        "llm_prediction": "Benign",
        "final_prediction": "Benign",
        "used_rag": False,
    }

    class Chain:
        def invoke(self, input_data, config=None):
            return expected.copy()

    monkeypatch.setattr(main, "JinaEmbedding", object)
    monkeypatch.setattr(main, "load_or_create_vector_store", lambda *args: object())
    monkeypatch.setattr(main, "create_llm", object)
    monkeypatch.setattr(main, "get_xgb_reranker", object)
    monkeypatch.setattr(main, "create_conversational_chain", lambda *args, **kwargs: Chain())

    result = main.analyze_contract(query="Analyze", system_input_text="{}")

    assert result["final_prediction"] == "Benign"
    assert "error" not in result


def test_feature_extraction_handles_deep_traces_without_recursion(monkeypatch):
    root = {"func": "0xa9059cbb", "internal_calls": []}
    current = root
    for _ in range(1_200):
        child = {"func": "0x095ea7b3", "internal_calls": []}
        current["internal_calls"].append(child)
        current = child

    monkeypatch.setattr(
        "txshield.detection.features.extractor.json.loads",
        lambda _: {"function_call": [root]},
    )
    features = extract_features_from_json_v2("{}")

    assert features[0] == 1_201
    assert features[2] == 1_201
    assert features[4] == 1
    assert features[5] == 1


def test_xgboost_receives_ten_features_after_four_are_removed():
    class Model:
        def predict_proba(self, values):
            assert values.shape == (1, 10)
            return np.asarray([[0.25, 0.75]])

    reranker = object.__new__(XGBoostReranker)
    reranker.model = Model()

    assert len(RAW_FEATURE_NAMES) == 14
    assert MODEL_DROP_INDICES == (7, 8, 9, 11)
    assert len(MODEL_FEATURE_NAMES) == 10
    assert reranker.predict_proba('{"function_call": []}') == 0.75
