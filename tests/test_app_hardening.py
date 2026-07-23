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

    def test_analyze_event_is_a_visible_named_api_endpoint(self) -> None:
        analyze_dependency = next(
            dependency
            for dependency in app.demo.config["dependencies"]
            if dependency.get("api_name") == "analyze"
        )

        self.assertEqual(analyze_dependency["api_visibility"], "public")
        self.assertIsNot(analyze_dependency.get("show_api"), False)

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

    def test_clears_key_and_logs_classifier_failure_without_request_data(self) -> None:
        validation_outputs = app.analyze_article("   ", self.api_key)
        self.assertIn("Enter a headline or article", validation_outputs[0])
        self.assert_key_cleared(validation_outputs)

        article = "private-classifier-article-7f6d"
        with (
            patch.object(
                app,
                "classify_with_uncertainty",
                side_effect=RuntimeError(f"failure included {article} {self.api_key}"),
            ),
            self.assertLogs(app.LOGGER, level="ERROR") as captured_logs,
        ):
            exception_outputs = app.analyze_article(article, self.api_key)

        logs = "\n".join(captured_logs.output)
        self.assertIn("Article analysis failed (RuntimeError)", logs)
        self.assertNotIn(article, logs)
        self.assertNotIn(self.api_key, logs)
        self.assertNotIn(article, "".join(exception_outputs))
        self.assertIn("could not be completed", exception_outputs[0])
        self.assert_key_cleared(exception_outputs)

    def test_retrieval_failure_is_private_and_keeps_the_classifier_result(self) -> None:
        article = "private-retrieval-article-2a91"
        with (
            patch.object(
                app,
                "classify_with_uncertainty",
                return_value=("real", 0.9, 0.02),
            ),
            patch.object(
                app,
                "verify_claim",
                side_effect=RuntimeError(f"retrieval included {article} {self.api_key}"),
            ),
            patch.object(
                app,
                "explain_decision",
                side_effect=app.AnthropicKeyRejectedError("key rejected"),
            ),
            self.assertLogs(app.LOGGER, level="ERROR") as captured_logs,
        ):
            outputs = app.analyze_article(article, self.api_key)

        logs = "\n".join(captured_logs.output)
        self.assertIn("Verification failed (RuntimeError)", logs)
        self.assertNotIn(article, logs)
        self.assertNotIn(self.api_key, logs)
        self.assertNotIn(article, "".join(outputs))
        self.assertIn("Final classifier label", outputs[0])
        self.assertIn("Verification was unavailable for this request", outputs[1])
        self.assertIn(app.REJECTED_KEY_MESSAGE, outputs[2])
        self.assert_key_cleared(outputs)

    def test_optional_explanation_failure_keeps_core_analysis(self) -> None:
        article = "private-explanation-article-93bc"
        with (
            patch.object(
                app,
                "classify_with_uncertainty",
                return_value=("real", 0.9, 0.02),
            ),
            patch.object(app, "verify_claim", return_value=self.verification_result()),
            patch.object(
                app,
                "explain_decision",
                side_effect=ValueError(f"provider failed for {article} {self.api_key}"),
            ),
            self.assertLogs(app.LOGGER, level="ERROR") as captured_logs,
        ):
            outputs = app.analyze_article(article, self.api_key)

        logs = "\n".join(captured_logs.output)
        self.assertIn("Claude explanation failed (ValueError)", logs)
        self.assertNotIn(article, logs)
        self.assertNotIn(self.api_key, logs)
        self.assertNotIn(article, "".join(outputs))
        self.assertIn("Final classifier label", outputs[0])
        self.assertIn("Claude service could not complete", outputs[2])
        self.assert_key_cleared(outputs)

    def test_malformed_evidence_url_does_not_abort_analysis(self) -> None:
        verification = self.verification_result()
        verification["evidence"] = [
            {
                "title": "Retrieved source",
                "snippet": "A short supporting excerpt.",
                "url": "https://[malformed",
                "nli": {"entailment": 0.8, "contradiction": 0.1},
            }
        ]
        with (
            patch.object(
                app,
                "classify_with_uncertainty",
                return_value=("real", 0.9, 0.02),
            ),
            patch.object(app, "verify_claim", return_value=verification),
            patch.object(app, "explain_decision", return_value="Structured explanation."),
        ):
            outputs = app.analyze_article(
                "City officials approved a transit budget after a public hearing.", ""
            )

        self.assertIn("Final classifier label", outputs[0])
        self.assertIn("<strong>Retrieved source</strong>", outputs[1])
        self.assertNotIn("https://[malformed", outputs[1])
        self.assertEqual(outputs[3], "")


if __name__ == "__main__":
    unittest.main()
