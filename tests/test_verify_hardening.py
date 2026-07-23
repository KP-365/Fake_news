"""Regression tests for private verification failures."""

from __future__ import annotations

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import verify


class VerifyHardeningTests(unittest.TestCase):
    def test_retrieval_error_does_not_log_or_print_claim_text(self) -> None:
        claim = "private-retrieval-claim-5c82"
        output = StringIO()

        with (
            patch.object(
                verify,
                "_search_ddg_with_hard_timeout",
                side_effect=RuntimeError(f"request failed for {claim}"),
            ),
            self.assertLogs(verify.LOGGER, level="WARNING") as captured_logs,
            redirect_stdout(output),
        ):
            evidence = verify.retrieve_evidence(claim)

        logs = "\n".join(captured_logs.output)
        self.assertEqual(evidence, [])
        self.assertIn("Evidence retrieval failed (RuntimeError)", logs)
        self.assertNotIn(claim, logs)
        self.assertNotIn(claim, output.getvalue())


if __name__ == "__main__":
    unittest.main()
