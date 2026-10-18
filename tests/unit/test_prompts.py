from txshield.contract_analyzer.core.message_manager import MessageManager


def test_rag_prompt_uses_retrieved_cases_without_xgboost_gating():
    messages = MessageManager.build_messages_with_prompt_template(
        query="Analyze the transaction.",
        system_input='{"calls": []}',
        reference_text="Historical case 1: reentrancy pattern",
        xgb_prob=0.99,
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "historical attack cases" in messages[0]["content"]
    assert "XGBoost" not in messages[0]["content"]
    assert "0.99" not in messages[0]["content"]
    assert "Historical case 1" in messages[1]["content"]
    assert "Comparative Call Sequence Analysis" in messages[0]["content"]
    assert "write only `Attack` or `Benign`" in messages[0]["content"]


def test_zero_shot_prompt_is_conservative_and_excludes_context():
    messages = MessageManager.build_messages_with_prompt_template(
        query="Analyze the transaction.",
        system_input='{"calls": []}',
        reference_text="Historical case 1: must not be used",
        skip_rag=True,
    )

    assert "No\nhistorical case passed" in messages[0]["content"]
    assert "In the absence of verifiable exploit behavior, choose\nBenign" in messages[0]["content"]
    assert "Historical case 1" not in messages[1]["content"]
    assert "Call Sequence Analysis" in messages[0]["content"]
