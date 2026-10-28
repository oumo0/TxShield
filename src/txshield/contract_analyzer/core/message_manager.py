from __future__ import annotations

class MessageManager:
    """Build the manuscript's retrieval-augmented and zero-shot prompts."""

    @staticmethod
    def build_messages_with_prompt_template(
        query: str,
        system_input: str,
        reference_text: str = "",
        skip_rag: bool = False,
        prompt_type: str = "rag",
        xgb_prob: float | None = None,
        xgb_context_threshold: float | None = None,
    ) -> list[dict[str, str]]:
        """Return Figure 2 or Figure 3 while retaining legacy parameters."""
        del query, prompt_type
        use_rag = not skip_rag and bool(reference_text.strip())
        prompt = (
            MessageManager._rag_prompt(
                system_input,
                reference_text,
                xgb_prob,
                xgb_context_threshold,
            )
            if use_rag
            else MessageManager._zero_shot_prompt(
                system_input,
                xgb_prob,
                xgb_context_threshold,
            )
        )
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def _risk_hint(
        xgb_prob: float | None,
        xgb_context_threshold: float | None,
    ) -> str:
        if xgb_prob is None:
            return ""
        threshold = 0.5 if xgb_context_threshold is None else xgb_context_threshold
        return f"""

When an XGBoost score is available:
XGBoost attack probability: {float(xgb_prob):.3f} (threshold {float(threshold):g}).
Use this as a bias corrector: if probability < {float(threshold):g}, you must pay extra attention to avoid false positives, but it does not override your own logic. Final decision must be justified by the transaction evidence."""

    @staticmethod
    def _rag_prompt(
        system_input: str,
        reference_text: str,
        xgb_prob: float | None,
        xgb_context_threshold: float | None,
    ) -> str:
        risk_hint = MessageManager._risk_hint(xgb_prob, xgb_context_threshold)
        return f"""Input:
* Current transaction: {system_input}
* Historical attack examples: {reference_text}

You are an expert in smart contract attack detection.
Your task is to analyze the Current transaction against Historical attack examples provided in the context to determine malicious intent.
The historical examples were retrieved due to high behavioral similarity to the current transaction. Therefore, you must assume malicious intent initially and focus on validating the attack pattern.

Follow the instructions:
1. Start assuming this is an Attack executing a known pattern.
2. Compare execution flow and fund transfers with historical examples.
3. Ignore function names; focus on operational sequence and security implications.
4. Focus on Inter-contract calls, abnormal fund movements, state manipulation patterns.{risk_hint}

Output MUST begin with a short answer 'Attack' or 'Benign', followed by the following information:
1. Behavior Summary: Describe contract interactions, fund movements, and final goal.
2. Comparative Call Sequence Analysis: Analyze call sequence. Compare with most similar historical ID(s). State shared malicious steps.
3. Malicious Indicators & Similarity Assessment: Identify attack-confirming behaviors. Rate similarity: High/Moderate/Low. If Benign, provide irrefutable evidence."""

    @staticmethod
    def _zero_shot_prompt(
        system_input: str,
        xgb_prob: float | None,
        xgb_context_threshold: float | None,
    ) -> str:
        risk_hint = MessageManager._risk_hint(xgb_prob, xgb_context_threshold)
        return f"""Input:
* Current transaction: {system_input}

You are an expert in smart contract attack detection.
Evaluate whether the current transaction is malicious based only on the provided transaction data.
No historical attack examples are provided. Follow these instructions:
1. Your judgment must be based purely on the internal logic, security best practices, and execution behavior of the current transaction.
2. Assume the transaction is Benign unless verifiable security flaws or abnormal logic are evident in the input.{risk_hint}

Output MUST begin with a short answer 'Attack' or 'Benign', followed by the following information:
1. Behavior Summary: Describe the transaction's overall behavior, including contract interactions and fund movements.
2. Call Sequence Analysis: Analyze the function call sequence. Highlight unusual operations or patterns violating smart contract security practices (e.g., state change before external call).
3. Malicious Indicators: Identify behaviors indicating an attack (e.g., reentrancy, unprivileged function calls, unauthorized balance changes) OR explain why the transaction is logically sound (if Benign)."""
