"""Evaluate NLI escalation on the least-confident WELFake test articles."""

from __future__ import annotations

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

from predict import ID_TO_LABEL, MAX_LENGTH, load_model
from verify import verify_claim

DATASET_NAME = "saurabhshahane/fake-news-classification"
MC_PASSES = 30
BATCH_SIZE = 32
BUCKET_SIZE = 100
REQUEST_DELAY_SECONDS = 2
SNIPPET_LENGTH = 300
RESULTS_PATH = Path(__file__).with_name("escalation_results.csv")
VERDICT_TO_LABEL = {"refuted": "fake", "supported": "real"}


def clean_text(text: object) -> str:
    """Apply the same text cleaning used by the training notebook."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def load_test_split() -> pd.DataFrame:
    """Recreate the notebook's deterministic, stratified WELFake test split."""
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


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """Enable only dropout layers while the rest of the model remains in eval mode."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_predictions(
    texts: list[str], model: torch.nn.Module, tokenizer: object, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return labels and confidence from mean probabilities over MC passes."""
    model.eval()
    enable_mc_dropout(model)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    predicted_labels: list[np.ndarray] = []
    confidences: list[np.ndarray] = []

    for batch_number, start in enumerate(
        range(0, len(texts), BATCH_SIZE), start=1
    ):
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
            print(
                f"MC Dropout: {completed:,}/{len(texts):,} articles", flush=True
            )
    return np.concatenate(predicted_labels), np.concatenate(confidences)


def verify_low_confidence_bucket(
    bucket: pd.DataFrame, classifier_labels: np.ndarray
) -> pd.DataFrame:
    """Run evidence retrieval and NLI on each selected article."""
    rows = []
    for position, ((_, article), classifier_label) in enumerate(
        zip(bucket.iterrows(), classifier_labels, strict=True), start=1
    ):
        claim = article["title"] or article["content"][:SNIPPET_LENGTH]
        try:
            result = verify_claim(claim)
            verdict = result["verdict"]
        except Exception as error:
            print(f"Verification {position}/{len(bucket)} failed: {error}", file=sys.stderr)
            verdict = "insufficient"

        final_label = VERDICT_TO_LABEL.get(verdict, classifier_label)
        rows.append(
            {
                "text_snippet": article["content"][:SNIPPET_LENGTH],
                "true_label": ID_TO_LABEL[int(article["label"])],
                "classifier_label": classifier_label,
                "verdict": verdict,
                "final_label": final_label,
            }
        )
        print(f"Verified {position}/{len(bucket)}: {verdict}")
        if position < len(bucket):
            time.sleep(REQUEST_DELAY_SECONDS)

    return pd.DataFrame(rows)


def main() -> None:
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

    results = verify_low_confidence_bucket(bucket, bucket_classifier_labels)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    classifier_accuracy = (results["classifier_label"] == results["true_label"]).mean()
    escalated_accuracy = (results["final_label"] == results["true_label"]).mean()
    verdict_distribution = Counter(results["verdict"])

    print(f"\nSaved {len(results)} rows to {RESULTS_PATH}")
    print(f"Classifier-only accuracy: {classifier_accuracy:.4f}")
    print(f"Classifier + NLI accuracy: {escalated_accuracy:.4f}")
    print("Verdict distribution:")
    for verdict in ("supported", "refuted", "insufficient"):
        print(f"  {verdict}: {verdict_distribution[verdict]}")


if __name__ == "__main__":
    main()
