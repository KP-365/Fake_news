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

    def request_explanation(self) -> tuple[Mock, Mock, str]:
        client = Mock()
        client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Plain-English explanation.")]
        )
        with patch("anthropic.Anthropic", return_value=client) as client_class:
            result = explain.explain_decision(
                classifier_label=self.signals["classifier_label"],
                confidence=self.signals["classifier_confidence"],
                mc_uncertainty=self.signals["mc_dropout_uncertainty"],
                nli_verdict=self.signals["nli_verdict"],
                max_entailment=self.signals["max_entailment_score"],
                max_contradiction=self.signals["max_contradiction_score"],
                evidence_count=self.signals["evidence_count"],
                api_key=self.api_key,
            )
        return client_class, client, result

    def test_prompt_requires_plain_english_before_each_technical_name(self) -> None:
        _, client, _ = self.request_explanation()
        request = client.messages.create.call_args.kwargs
        system_prompt = " ".join(request["system"].split())
        user_prompt = " ".join(request["messages"][0]["content"].split())
        required_descriptions = (
            "how sure the classifier was [confidence]",
            "how consistent the classifier was across repeated runs "
            "[Monte Carlo dropout uncertainty]",
            "how well the retrieved evidence agreed with the claim [NLI]",
        )

        self.assertIn("plain-English meaning", system_prompt)
        for description in required_descriptions:
            self.assertIn(description, system_prompt)
            self.assertIn(description, user_prompt)

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
        self.assertEqual(result, "Plain-English explanation.")
        client_class.assert_called_once_with(api_key=self.api_key)
        self.assertNotIn(self.api_key, serialized_request)
        self.assertNotIn("article_text", serialized_request)
        self.assertNotIn("evidence_text", serialized_request)


if __name__ == "__main__":
    unittest.main()
