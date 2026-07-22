"""Compute report statistics from the committed evaluation CSV files."""

from __future__ import annotations

import csv
import math
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist

EVALUATION_DIR = Path(__file__).resolve().parent
ESCALATION_PATH = EVALUATION_DIR / "escalation_results.csv"
FAITHFULNESS_PATH = EVALUATION_DIR / "faithfulness_review.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV into dictionaries and reject empty datasets."""
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return rows


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial proportion."""
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("Expected 0 <= successes <= total and total > 0")

    z_score = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = successes / total
    z_squared = z_score**2
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    half_width = (
        z_score
        * math.sqrt(
            (proportion * (1 - proportion) + z_squared / (4 * total)) / total
        )
        / denominator
    )
    return center - half_width, center + half_width


def exact_mcnemar_p_value(fixed: int, broken: int) -> Fraction:
    """Return the conventional two-sided exact McNemar binomial p-value."""
    if fixed < 0 or broken < 0 or fixed + broken == 0:
        raise ValueError("McNemar discordant counts must be nonnegative and nonzero")

    discordant = fixed + broken
    smaller_count = min(fixed, broken)
    lower_tail_combinations = sum(
        math.comb(discordant, count) for count in range(smaller_count + 1)
    )
    return min(Fraction(1), Fraction(2 * lower_tail_combinations, 2**discordant))


def percent(value: float) -> str:
    """Format a proportion as a percentage with one decimal place."""
    return f"{100 * value:.1f}%"


def main() -> None:
    escalation_rows = read_csv(ESCALATION_PATH)
    total = len(escalation_rows)
    classifier_correct = sum(
        row["classifier_label"] == row["true_label"] for row in escalation_rows
    )
    final_correct = sum(
        row["final_label"] == row["true_label"] for row in escalation_rows
    )
    fixed = sum(
        row["classifier_label"] != row["true_label"]
        and row["final_label"] == row["true_label"]
        for row in escalation_rows
    )
    broken = sum(
        row["classifier_label"] == row["true_label"]
        and row["final_label"] != row["true_label"]
        for row in escalation_rows
    )

    faithfulness_rows = read_csv(FAITHFULNESS_PATH)
    faithful = sum(
        row["matches_signals_yes_no"].strip().casefold() == "yes"
        for row in faithfulness_rows
    )
    faithfulness_total = len(faithfulness_rows)

    expected = (100, 76, 64, 11, 23, 10, 8)
    observed = (
        total,
        classifier_correct,
        final_correct,
        fixed,
        broken,
        faithfulness_total,
        faithful,
    )
    if observed != expected:
        raise ValueError(f"Committed CSV counts changed: expected {expected}, got {observed}")

    classifier_ci = wilson_interval(classifier_correct, total)
    final_ci = wilson_interval(final_correct, total)
    mcnemar_p = exact_mcnemar_p_value(fixed, broken)
    faithfulness_ci = wilson_interval(faithful, faithfulness_total)

    print(
        f"Baseline classifier accuracy was {percent(classifier_correct / total)} "
        f"({classifier_correct}/{total}; 95% Wilson CI, "
        f"{percent(classifier_ci[0])}–{percent(classifier_ci[1])})."
    )
    print(
        f"Post-override accuracy was {percent(final_correct / total)} "
        f"({final_correct}/{total}; 95% Wilson CI, "
        f"{percent(final_ci[0])}–{percent(final_ci[1])})."
    )
    print(
        f"Overrides fixed {fixed} errors and broke {broken} correct predictions; "
        f"the two-sided exact McNemar test gave p = {float(mcnemar_p):.6f}."
    )
    print(
        f"Explanation faithfulness was {percent(faithful / faithfulness_total)} "
        f"({faithful}/{faithfulness_total}; 95% Wilson CI, "
        f"{percent(faithfulness_ci[0])}–{percent(faithfulness_ci[1])})."
    )


if __name__ == "__main__":
    main()
