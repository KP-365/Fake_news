"""Run Google-first verification on the committed 100-row escalation bucket."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARENT_REPO_ROOT = PROJECT_ROOT.parent
SOURCE_RESULTS_PATH = PARENT_REPO_ROOT / "evaluation" / "escalation_results.csv"
RESULTS_PATH = Path(__file__).resolve().parent / "google_escalation_results.csv"
SUMMARY_PATH = Path(__file__).resolve().parent / "google_escalation_summary.txt"
RATE_LIMIT_SECONDS = 2.0
EXPECTED_BUCKET_SIZE = 100
EXPECTED_CLASSIFIER_CORRECT = 76
EXPECTED_DDG_CORRECT = 64
GOOGLE_KEY_ENV_NAMES = (
    "GOOGLE_FACTCHECK_API_KEY",
    "GOOGLE_FACT_CHECK_API_KEY",
)
VERDICT_TO_LABEL = {"supported": "real", "refuted": "fake"}

# This runner never uses the classifier GPU path. If DeBERTa fallback is needed,
# verify.py will select MPS when available and otherwise CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _configure_google_key() -> str:
    """Validate either supported key name and normalize it for verify.py."""
    for environment_name in GOOGLE_KEY_ENV_NAMES:
        api_key = os.getenv(environment_name, "").strip()
        if api_key:
            os.environ["GOOGLE_FACTCHECK_API_KEY"] = api_key
            return environment_name
    names = " or ".join(GOOGLE_KEY_ENV_NAMES)
    raise RuntimeError(
        f"Google Fact Check API key is missing. Set {names} before running "
        "the fixed-bucket evaluation."
    )


def _load_verify_claim() -> Callable[[str], dict[str, Any]]:
    """Import this folder's verifier only after key validation succeeds."""
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from verify import verify_claim

    return verify_claim


def _load_fixed_bucket() -> tuple[pd.DataFrame, str]:
    """Load and validate the committed DDG escalation bucket."""
    if not SOURCE_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Committed escalation bucket not found at {SOURCE_RESULTS_PATH}"
        )

    bucket = pd.read_csv(SOURCE_RESULTS_PATH)
    claim_column = "claim" if "claim" in bucket.columns else "text_snippet"
    required_columns = {
        claim_column,
        "classifier_label",
        "true_label",
        "final_label",
    }
    missing_columns = required_columns.difference(bucket.columns)
    if missing_columns:
        raise ValueError(
            f"Escalation bucket is missing columns: {sorted(missing_columns)}"
        )
    if len(bucket) != EXPECTED_BUCKET_SIZE:
        raise ValueError(
            f"Expected the committed {EXPECTED_BUCKET_SIZE}-row bucket, "
            f"found {len(bucket)} rows"
        )
    if bucket[claim_column].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError(f"Escalation bucket contains an empty {claim_column}")

    classifier_correct = int(
        (bucket["classifier_label"] == bucket["true_label"]).sum()
    )
    ddg_correct = int((bucket["final_label"] == bucket["true_label"]).sum())
    if classifier_correct != EXPECTED_CLASSIFIER_CORRECT:
        raise ValueError(
            "Committed classifier baseline changed: expected "
            f"{EXPECTED_CLASSIFIER_CORRECT}/{EXPECTED_BUCKET_SIZE}, got "
            f"{classifier_correct}/{len(bucket)}"
        )
    if ddg_correct != EXPECTED_DDG_CORRECT:
        raise ValueError(
            "Committed DDG-only final baseline changed: expected "
            f"{EXPECTED_DDG_CORRECT}/{EXPECTED_BUCKET_SIZE}, got "
            f"{ddg_correct}/{len(bucket)}"
        )
    return bucket, claim_column


def _hybrid_final_label(verdict: str, classifier_label: str) -> str:
    return VERDICT_TO_LABEL.get(verdict, classifier_label)


def _build_summary(results: pd.DataFrame) -> str:
    total = len(results)
    google_covered = results["google_candidate_count"] > 0
    google_usable = (
        (results["source"] == "google_fact_check")
        & results["verdict"].isin(VERDICT_TO_LABEL)
    )
    google_rows = results.loc[google_usable]
    google_agreement = (
        google_rows["verdict"].map(VERDICT_TO_LABEL)
        == google_rows["true_label"]
    )

    classifier_correct = int(
        (results["classifier_label"] == results["true_label"]).sum()
    )
    ddg_correct = int(
        (
            results["committed_ddg_final_label"]
            == results["true_label"]
        ).sum()
    )
    hybrid_correct = int((results["final_label"] == results["true_label"]).sum())

    lines = [
        "Google fixed-bucket summary",
        "===========================",
        f"Rows: {total}",
        (
            "Google coverage rate: "
            f"{google_covered.mean():.1%} "
            f"({int(google_covered.sum())}/{total})"
        ),
        (
            "Usable-verdict rate: "
            f"{google_usable.mean():.1%} "
            f"({int(google_usable.sum())}/{total})"
        ),
    ]
    if len(google_rows):
        lines.append(
            "Google agreement with true labels: "
            f"{google_agreement.mean():.1%} "
            f"({int(google_agreement.sum())}/{len(google_rows)})"
        )
    else:
        lines.append("Google agreement with true labels: n/a (0/0)")

    lines.extend(
        [
            "Accuracy comparison:",
            (
                "  Classifier-only: "
                f"{classifier_correct / total:.1%} "
                f"({classifier_correct}/{total})"
            ),
            (
                "  Committed DDG-only final: "
                f"{ddg_correct / total:.1%} ({ddg_correct}/{total})"
            ),
            (
                "  Google-first hybrid final: "
                f"{hybrid_correct / total:.1%} ({hybrid_correct}/{total})"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def run_bucket() -> pd.DataFrame:
    """Verify the frozen bucket and persist row-level and summary outputs."""
    _configure_google_key()
    verify_claim = _load_verify_claim()
    bucket, claim_column = _load_fixed_bucket()

    records: list[dict[str, Any]] = []
    for position, (row_id, row) in enumerate(bucket.iterrows(), start=1):
        claim = str(row[claim_column]).strip()
        verification = verify_claim(claim)
        verdict = str(verification.get("verdict") or "insufficient")
        source = str(verification.get("source") or "none")
        google_candidates = verification.get("google_candidates") or []
        classifier_label = str(row["classifier_label"])

        records.append(
            {
                "row_id": int(row_id),
                "claim": claim,
                "true_label": str(row["true_label"]),
                "classifier_label": classifier_label,
                "committed_ddg_final_label": str(row["final_label"]),
                "google_candidate_count": len(google_candidates),
                "source": source,
                "verdict": verdict,
                "final_label": _hybrid_final_label(
                    verdict, classifier_label
                ),
            }
        )

        results = pd.DataFrame(records)
        results.to_csv(RESULTS_PATH, index=False)
        print(
            f"[{position:03d}/{len(bucket)}] row={row_id} "
            f"google_candidates={len(google_candidates)} "
            f"source={source} verdict={verdict}",
            flush=True,
        )
        if position < len(bucket):
            time.sleep(RATE_LIMIT_SECONDS)

    summary = _build_summary(results)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(f"\nSaved {len(results)} rows to {RESULTS_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}\n")
    print(summary, end="")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Google-first verification on the committed 100-row "
            "escalation bucket without rerunning classification."
        )
    )
    parser.add_argument(
        "--check-key-only",
        action="store_true",
        help="validate key configuration and exit before imports or requests",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    key_source = _configure_google_key()
    if args.check_key_only:
        print(
            f"Google key configuration is valid via {key_source}; "
            "no request was sent."
        )
        return 0

    run_bucket()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
