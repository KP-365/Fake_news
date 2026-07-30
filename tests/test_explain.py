"""Focused tests for Claude's structured-signals-only prompt contract."""

from __future__ import annotations

import inspect
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import explain


class ExplanationPromptTests(unittest.TestCase):
    api_key = "private-key-sentinel-b61a"
    signals = {
        "classifier_label": "real",
        "classifier_confidence": 0.91,
        "mc_dropout_uncertainty": 0.04,
        "nli_verdict": "supported",
        "max_entailment_score": 0.82,
        "max_contradiction_score": 0.07,
        "evidence_count": 4,
    }

    def request_explanation(self, **overrides: object) -> tuple[Mock, Mock, str]:
        signals = {**self.signals, **overrides}
        explanation_text = (
            f"The system gave the final label {signals['classifier_confidence']:.1%} "
            "confidence, meaning this was its score rather than the chance of being correct. "
            f"Repeated checks varied by {signals['mc_dropout_uncertainty'] * 100:.1f} "
            "percentage points, meaning smaller variation is more consistent. "
            f"The strongest supporting match was {signals['max_entailment_score']:.1%}, "
            "meaning one passage supported the claim most strongly, while the strongest "
            f"conflicting match was {signals['max_contradiction_score']:.1%}, meaning one "
            "passage conflicted most strongly."
        )
        client = Mock()
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=explanation_text)]
        )
        with patch("anthropic.Anthropic", return_value=client) as client_class:
            result = explain.explain_decision(
                classifier_label=signals["classifier_label"],
                confidence=signals["classifier_confidence"],
                mc_uncertainty=signals["mc_dropout_uncertainty"],
                nli_verdict=signals["nli_verdict"],
                max_entailment=signals["max_entailment_score"],
                max_contradiction=signals["max_contradiction_score"],
                evidence_count=signals["evidence_count"],
                api_key=self.api_key,
            )
        return client_class, client, result

    def test_supported_evidence_uses_safe_percentages_with_immediate_meanings(self) -> None:
        _, client, result = self.request_explanation()
        request = client.messages.create.call_args.kwargs
        system_prompt = " ".join(request["system"].split())
        user_prompt = " ".join(request["messages"][0]["content"].split())

        self.assertIn("exactly one concise paragraph", system_prompt)
        self.assertIn(
            "91.0% confidence: the fixed system run's strength of preference",
            user_prompt,
        )
        self.assertIn(
            "4.0 percentage points repeated-check variation: this shows how much the fake "
            "score varied across 30 checks",
            user_prompt,
        )
        self.assertIn(
            "82.0% strongest supporting match",
            user_prompt,
        )
        self.assertIn(
            "7.0% strongest conflicting match",
            user_prompt,
        )
        self.assertIn(
            "The evidence verdict was supported, meaning at least one retrieved passage "
            "supported the claim",
            user_prompt,
        )
        for forbidden_term in explain.FORBIDDEN_OUTPUT_TERMS:
            self.assertNotIn(forbidden_term, result.casefold())

    def test_conflicting_evidence_is_context_not_an_override(self) -> None:
        _, client, _ = self.request_explanation(
            nli_verdict="refuted",
            max_entailment_score=0.12,
            max_contradiction_score=0.86,
        )
        user_prompt = " ".join(
            client.messages.create.call_args.kwargs["messages"][0]["content"].split()
        )

        self.assertIn("86.0% strongest conflicting match", user_prompt)
        self.assertIn(
            "The evidence verdict was refuted, meaning at least one retrieved passage "
            "conflicted with the claim",
            user_prompt,
        )
        self.assertIn(
            "This conflicts with the fixed label; state immediately that it is context only "
            "and does not change that label",
            user_prompt,
        )

    def test_insufficient_evidence_does_not_turn_scores_into_a_conclusion(self) -> None:
        _, client, _ = self.request_explanation(
            nli_verdict="insufficient",
            max_entailment_score=0.41,
            max_contradiction_score=0.34,
        )
        user_prompt = " ".join(
            client.messages.create.call_args.kwargs["messages"][0]["content"].split()
        )

        self.assertIn(
            "The evidence verdict was insufficient: the evidence checks did not resolve the "
            "claim",
            user_prompt,
        )
        self.assertIn(
            "Do not treat low or zero scores as affirmative evidence", user_prompt
        )

    def test_request_sends_only_structured_signals_not_private_text_or_key(self) -> None:
        client_class, client, result = self.request_explanation()
        request = client.messages.create.call_args.kwargs
        user_prompt = request["messages"][0]["content"]
        signal_payload = json.loads(user_prompt.split("Structured signals:\n", 1)[1])
        serialized_request = json.dumps(request)
        parameters = inspect.signature(explain.explain_decision).parameters

        self.assertNotIn("article_text", parameters)
        self.assertNotIn("evidence_text", parameters)
        self.assertEqual(signal_payload, self.signals)
        self.assertEqual(result, client.messages.create.return_value.content[0].text)
        client_class.assert_called_once_with(api_key=self.api_key)
        self.assertNotIn(self.api_key, serialized_request)
        self.assertNotIn("article_text", serialized_request)
        self.assertNotIn("evidence_text", serialized_request)


class ExplanationOutputValidationTests(unittest.TestCase):
    metrics = {
        "confidence": 0.91,
        "mc_uncertainty": 0.04,
        "max_entailment": 0.82,
        "max_contradiction": 0.07,
    }
    valid_explanation = (
        "The final label had 91.0% confidence, meaning the system preferred that label. "
        "Repeated checks varied by 4.0 percentage points, meaning the result was consistent. "
        "The strongest supporting match was 82.0%, meaning one passage supported the claim, "
        "and the strongest conflicting match was 7.0%, meaning another passage conflicted."
    )

    def test_accepts_one_plain_paragraph_with_every_display_value(self) -> None:
        result = explain._validate_explanation_output(
            self.valid_explanation, **self.metrics
        )

        self.assertEqual(result, self.valid_explanation)

    def test_rejects_multiple_paragraphs(self) -> None:
        explanation = self.valid_explanation.replace(" Repeated", "\n\nRepeated")
        with self.assertRaisesRegex(RuntimeError, "exactly one paragraph"):
            explain._validate_explanation_output(explanation, **self.metrics)

    def test_rejects_a_missing_display_value(self) -> None:
        explanation = self.valid_explanation.replace("82.0%", "the available score")
        with self.assertRaisesRegex(RuntimeError, "82.0%"):
            explain._validate_explanation_output(explanation, **self.metrics)

    def test_rejects_forbidden_technical_terms(self) -> None:
        explanation = self.valid_explanation + " It came from dropout checks."
        with self.assertRaisesRegex(RuntimeError, "forbidden technical term.*dropout"):
            explain._validate_explanation_output(explanation, **self.metrics)

if __name__ == "__main__":
    unittest.main()
