from txshield.preprocessing.pruning.rules import apply_general_rules

def test_duplicate_node_removed():
    assert apply_general_rules({"_duplicate": True}) is None

def test_args_shortened():
    node = {"args": "x" * 10}
    result = apply_general_rules(node, config={"max_arg_length": 5})
    assert result["args"] == "xxxxx...<truncated>"
