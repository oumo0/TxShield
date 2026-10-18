import json
from txshield.detection.features.extractor import extract_features_from_json
from txshield.contract_analyzer.utils.features import extract_features_from_json_v2

def test_empty_json_returns_fourteen_features():
    assert len(extract_features_from_json("{}")) == 14

def test_transfer_selector_is_detected():
    payload = {"function_call": [{"function": "transfer", "contract": "0xabc", "value": "0"}]}
    features = extract_features_from_json(json.dumps(payload))
    assert features[0] == 1
    assert features[4] == 1


def test_training_and_runtime_use_identical_feature_semantics():
    payload = {
        "function_call": [{
            "function": "transfer",
            "contract": "0xabc",
            "value": "0x10",
            "internal_calls": [{
                "function": "approve",
                "contract": "0xdef",
                "value": "0",
            }],
        }]
    }
    encoded = json.dumps(payload)
    training_features = extract_features_from_json(encoded)
    runtime_features = extract_features_from_json_v2(encoded)
    assert training_features == runtime_features
    assert training_features[2] == 2
