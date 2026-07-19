"""Regression tests for request-size and API-key handling in the Gradio callback."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import app


class AppHardeningTests(unittest.TestCase):
    api_key = "sensitive-test-key-value"

    def assert_key_cleared(self, outputs: tuple[str, str, str, str]) -> None:
        self.assertEqual(len(outputs), 4)
        self.assertEqual(outputs[3], "")
        self.assertTrue(all(self.api_key not in output for output in outputs))

    @staticmethod
    def verification_result() -> dict:
        return {
            "verdict": "supported",
            "evidence": [],
            "max_entailment": 0.8,
            "max_contradiction": 0.1,
        }

    def test_truncates_article_and_redacts_key_from_all_outputs(self) -> None:
        captured_texts: list[str] = []

        def classify(text: str, **_: object) -> tuple[str, float, float]:
            captured_texts.append(text)
            return "real", 0.9, 0.02

        def explain(**kwargs: object) -> str:
            return f"Provider response echoed {kwargs['api_key']}"

        with (
            patch.object(app, "classify_with_uncertainty", side_effect=classify),
            patch.object(app, "verify_claim", return_value=self.verification_result()),
            patch.object(app, "explain_decision", side_effect=explain),
        ):
            outputs = app.analyze_article("x" * 10_000, self.api_key)

        self.assertEqual(len(captured_texts), 1)
        self.assertEqual(len(captured_texts[0]), app.MAX_ARTICLE_CHARS)
        self.assertIn("truncated to the first 5,000 characters", outputs[0])
        self.assert_key_cleared(outputs)

    def test_clears_key_for_validation_and_classifier_exceptions(self) -> None:
        validation_outputs = app.analyze_article("   ", self.api_key)
        self.assertIn("Enter a headline or article", validation_outputs[0])
        self.assert_key_cleared(validation_outputs)

        with patch.object(
            app,
            "classify_with_uncertainty",
            side_effect=RuntimeError(f"failure included {self.api_key}"),
        ):
            exception_outputs = app.analyze_article("article", self.api_key)

        self.assertIn("could not be completed", exception_outputs[0])
        self.assert_key_cleared(exception_outputs)

    def test_clears_key_for_retrieval_and_rejected_key_paths(self) -> None:
        with (
            patch.object(
                app,
                "classify_with_uncertainty",
                return_value=("real", 0.9, 0.02),
            ),
            patch.object(
                app,
                "verify_claim",
                side_effect=RuntimeError(f"retrieval included {self.api_key}"),
            ),
            patch.object(
                app,
                "explain_decision",
                side_effect=app.AnthropicKeyRejectedError("key rejected"),
            ),
        ):
            outputs = app.analyze_article("article", self.api_key)

        self.assertIn(app.REJECTED_KEY_MESSAGE, outputs[2])
        self.assert_key_cleared(outputs)


if __name__ == "__main__":
    unittest.main()
