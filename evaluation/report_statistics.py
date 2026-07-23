"""Compute report statistics from committed evaluation artifacts."""

from __future__ import annotations

import csv
import math
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist

import numpy as np

EVALUATION_DIR = Path(__file__).resolve().parent
ESCALATION_PATH = EVALUATION_DIR / "escalation_results.csv"
FAITHFULNESS_PATH = EVALUATION_DIR / "faithfulness_review.csv"
MC_ARRAYS_DIR = EVALUATION_DIR / "mc_arrays"
PREDICTIVE_ENTROPY_PATH = MC_ARRAYS_DIR / "predictive_entropy.npy"
MEAN_PROBABILITIES_PATH = MC_ARRAYS_DIR / "mean_probabilities.npy"
TRUE_LABELS_PATH = MC_ARRAYS_DIR / "true_labels.npy"


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


def rejection_curve_point(deferred_count: int) -> tuple[int, int, int]:
    """Return total retained, total correct retained, and total test articles."""
    predictive_entropy = np.load(PREDICTIVE_ENTROPY_PATH, allow_pickle=False)
    mean_probabilities = np.load(MEAN_PROBABILITIES_PATH, allow_pickle=False)
    true_labels = np.load(TRUE_LABELS_PATH, allow_pickle=False)

    if predictive_entropy.ndim != 1 or true_labels.ndim != 1:
        raise ValueError("Predictive entropy and true labels must be one-dimensional")
    if mean_probabilities.ndim != 2 or mean_probabilities.shape[1] < 2:
        raise ValueError(
            "Mean probabilities must have one row per article and at least two classes"
        )

    total_count = true_labels.size
    if (
        predictive_entropy.size != total_count
        or mean_probabilities.shape[0] != total_count
    ):
        raise ValueError("MC arrays must contain the same number of articles")
    if not 0 < deferred_count < total_count:
        raise ValueError(
            "Deferred count must be greater than zero and smaller than the test set"
        )
    if (
        not np.isfinite(predictive_entropy).all()
        or not np.isfinite(mean_probabilities).all()
        or not np.isfinite(true_labels).all()
    ):
        raise ValueError("MC arrays contain non-finite values")

    retained_count = total_count - deferred_count
    retained_indices = np.argsort(predictive_entropy)[:retained_count]
    retained_predictions = mean_probabilities[retained_indices].argmax(axis=1)
    retained_correct = np.count_nonzero(
        retained_predictions == true_labels[retained_indices]
    )
    return retained_count, int(retained_correct), total_count


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

    deferred_count = 100
    retained_count, retained_correct, test_count = rejection_curve_point(deferred_count)
    print(
        f"After deferring the {deferred_count} highest-entropy articles, "
        f"{retained_count:,}/{test_count:,} were retained "
        f"({retained_count / test_count:.2%} coverage) with "
        f"{retained_correct:,}/{retained_count:,} correct "
        f"({retained_correct / retained_count:.2%} retained accuracy)."
    )


if __name__ == "__main__":
    main()
