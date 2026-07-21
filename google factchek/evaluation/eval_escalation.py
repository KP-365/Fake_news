"""Evaluate Google-first hybrid escalation on low-confidence WELFake articles."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predict import ID_TO_LABEL, MAX_LENGTH, enable_mc_dropout, load_model
from verify import verify_claim

GOOGLE_API_KEY_ENV = "GOOGLE_FACTCHECK_API_KEY"
DATASET_NAME = "saurabhshahane/fake-news-classification"
MC_PASSES = 30
BATCH_SIZE = 32
BUCKET_SIZE = 100
REQUEST_DELAY_SECONDS = 2
SNIPPET_LENGTH = 300
RESULTS_PATH = Path(__file__).with_name("escalation_results.csv")
VERDICT_TO_LABEL = {"refuted": "fake", "supported": "real"}


def clean_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_test_split() -> pd.DataFrame:
    dataset_path = Path(kagglehub.dataset_download(DATASET_NAME))
    welfake_df = pd.read_csv(dataset_path / "WELFake_Dataset.csv")

    if "Unnamed: 0" in welfake_df.columns:
        welfake_df = welfake_df.drop(columns=["Unnamed: 0"])

    welfake_df["title"] = welfake_df["title"].fillna("").apply(clean_text)
    welfake_df = welfake_df.dropna(subset=["text"])
    welfake_df["text"] = welfake_df["text"].apply(clean_text)
    welfake_df = welfake_df[welfake_df["text"].str.len() > 0]
    welfake_df["content"] = (
        welfake_df["title"] + ". " + welfake_df["text"]
    ).str.strip(". ")
    welfake_df = welfake_df.drop_duplicates(subset=["text"])

    _, temp_df = train_test_split(
        welfake_df,
        test_size=0.3,
        stratify=welfake_df["label"],
        random_state=42,
    )
    _, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=42,
    )
    return test_df.reset_index(drop=True)


def mc_dropout_predictions(
    texts: list[str], model: torch.nn.Module, tokenizer: object, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    enable_mc_dropout(model)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    predicted_labels: list[np.ndarray] = []
    confidences: list[np.ndarray] = []

    for batch_number, start in enumerate(range(0, len(texts), BATCH_SIZE), start=1):
        batch_texts = texts[start : start + BATCH_SIZE]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

        probability_sum = torch.zeros(
            (len(batch_texts), len(ID_TO_LABEL)), device=device
        )
        with torch.no_grad():
            for _ in range(MC_PASSES):
                probability_sum += torch.softmax(model(**encoded).logits, dim=-1)

        mean_probabilities = probability_sum / MC_PASSES
        batch_confidences, batch_predictions = mean_probabilities.max(dim=-1)
        predicted_labels.append(batch_predictions.cpu().numpy())
        confidences.append(batch_confidences.cpu().numpy())

        completed = min(start + BATCH_SIZE, len(texts))
        if batch_number % 10 == 0 or completed == len(texts):
            print(f"MC Dropout: {completed:,}/{len(texts):,} articles", flush=True)

    return np.concatenate(predicted_labels), np.concatenate(confidences)


def _google_fields(result: dict) -> dict:
    match = result.get("google_match") or {}
    return {
        "google_candidate_count": len(result.get("google_candidates") or []),
        "google_match_score": match.get("match_score"),
        "google_matched_claim": match.get("matched_claim", ""),
        "google_rating": match.get("textual_rating", ""),
        "google_publisher": match.get("publisher", ""),
        "google_review_url": match.get("review_url", ""),
        "google_reason": result.get("google_reason", result.get("reason", "")),
    }


def _ddg_fields(result: dict) -> dict:
    evidence = result.get("evidence") or []
    best = evidence[0] if evidence else {}
    best_nli = best.get("nli") or {}
    return {
        "max_entailment": result.get("max_entailment"),
        "max_contradiction": result.get("max_contradiction"),
        "best_evidence_title": best.get("title", ""),
        "best_evidence_url": best.get("url", ""),
        "best_evidence_snippet": best.get("snippet", ""),
        "best_entailment": best_nli.get("entailment"),
        "best_neutral": best_nli.get("neutral"),
        "best_contradiction": best_nli.get("contradiction"),
    }


def verify_low_confidence_bucket(
    bucket: pd.DataFrame,
    classifier_labels: np.ndarray,
    classifier_confidences: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for position, ((original_index, article), classifier_label, confidence) in enumerate(
        zip(
            bucket.iterrows(),
            classifier_labels,
            classifier_confidences,
            strict=True,
        ),
        start=1,
    ):
        claim = article["title"] or article["content"][:SNIPPET_LENGTH]
        try:
            result = verify_claim(claim)
        except Exception as error:
            print(
                f"Verification {position}/{len(bucket)} failed: {error}",
                file=sys.stderr,
            )
            result = {
                "verdict": "insufficient",
                "source": "none",
                "reason": f"verification exception: {type(error).__name__}: {error}",
            }

        verdict = result["verdict"]
        source = result.get("source", "none")
        final_label = VERDICT_TO_LABEL.get(verdict, classifier_label)

        row = {
            "test_index": int(original_index),
            "claim": claim,
            "text_snippet": article["content"][:SNIPPET_LENGTH],
            "true_label": ID_TO_LABEL[int(article["label"])],
            "classifier_label": classifier_label,
            "classifier_confidence": float(confidence),
            "verification_source": source,
            "verdict": verdict,
            "final_label": final_label,
            "label_changed": final_label != classifier_label,
            "verification_reason": result.get("reason", ""),
            **_google_fields(result),
            **_ddg_fields(result),
            "verification_payload_json": json.dumps(result, ensure_ascii=False),
        }
        rows.append(row)
        print(
            f"Verified {position}/{len(bucket)}: {source} -> {verdict} -> {final_label}"
        )
        if position < len(bucket):
            time.sleep(REQUEST_DELAY_SECONDS)

    return pd.DataFrame(rows)


def main() -> None:
    if not os.getenv(GOOGLE_API_KEY_ENV, "").strip():
        raise RuntimeError(
            f"{GOOGLE_API_KEY_ENV} is not set. Export the key (or store it in "
            "Colab Secrets and copy it into the environment) before running this "
            "evaluation; without it the Google tier silently returns zero "
            "candidates and the results are not interpretable."
        )

    test_df = load_test_split()
    print(f"Loaded {len(test_df):,} held-out WELFake test articles")

    model, tokenizer, device = load_model()
    if device.type == "cpu" and torch.backends.mps.is_available():
        device = torch.device("mps")
        model.to(device)
    print(f"Classifier device: {device}")

    predictions, confidences = mc_dropout_predictions(
        test_df["content"].tolist(), model, tokenizer, device
    )

    lowest_confidence_indices = np.argsort(confidences, kind="stable")[:BUCKET_SIZE]
    bucket = test_df.iloc[lowest_confidence_indices].copy()
    bucket_classifier_labels = np.array(
        [ID_TO_LABEL[int(label)] for label in predictions[lowest_confidence_indices]]
    )
    bucket_confidences = confidences[lowest_confidence_indices]

    results = verify_low_confidence_bucket(
        bucket,
        bucket_classifier_labels,
        bucket_confidences,
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    classifier_accuracy = (results["classifier_label"] == results["true_label"]).mean()
    escalated_accuracy = (results["final_label"] == results["true_label"]).mean()
    verdict_distribution = Counter(results["verdict"])
    source_distribution = Counter(results["verification_source"])

    google_covered = results["google_candidate_count"] > 0
    google_usable = results["verification_source"] == "google_fact_check"
    google_verdicts = results.loc[google_usable]
    google_agreement = (
        (google_verdicts["verdict"].map(VERDICT_TO_LABEL) == google_verdicts["true_label"]).mean()
        if len(google_verdicts)
        else float("nan")
    )

    print(f"\nSaved {len(results)} rows to {RESULTS_PATH}")
    print(f"Classifier-only accuracy: {classifier_accuracy:.4f}")
    print(f"Google + DDG/DeBERTa escalated accuracy: {escalated_accuracy:.4f}")
    print(f"Accuracy change: {escalated_accuracy - classifier_accuracy:+.4f}")
    print(f"Labels changed: {int(results['label_changed'].sum())}/{len(results)}")

    print("Google coverage:")
    print(
        f"  Coverage rate (>=1 candidate): {google_covered.mean():.4f} "
        f"({int(google_covered.sum())}/{len(results)})"
    )
    print(
        f"  Usable-verdict rate: {google_usable.mean():.4f} "
        f"({int(google_usable.sum())}/{len(results)})"
    )
    if len(google_verdicts):
        print(
            "  Google-verdict agreement with true labels: "
            f"{google_agreement:.4f} "
            f"({int((google_verdicts['verdict'].map(VERDICT_TO_LABEL) == google_verdicts['true_label']).sum())}"
            f"/{len(google_verdicts)})"
        )
    else:
        print("  Google-verdict agreement with true labels: n/a (no usable Google verdicts)")

    print("Verification source distribution:")
    for source in ("google_fact_check", "ddg_deberta", "none"):
        print(f"  {source}: {source_distribution[source]}")

    print("Verdict distribution:")
    for verdict in ("supported", "refuted", "insufficient"):
        print(f"  {verdict}: {verdict_distribution[verdict]}")


if __name__ == "__main__":
    main()
