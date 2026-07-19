"""Run classification, uncertainty estimation, verification, and explanation."""

from __future__ import annotations

import argparse

import torch
from dotenv import load_dotenv

from explain import explain_decision
from predict import ID_TO_LABEL, MAX_LENGTH, enable_mc_dropout, load_model
from verify import verify_claim

MC_PASSES = 30
FAKE_LABEL_ID = next(label_id for label_id, label in ID_TO_LABEL.items() if label == "fake")


def classify_with_uncertainty(article_text: str) -> tuple[str, float, float]:
    """Return the fixed classifier label, confidence, and fake-probability std."""
    model, tokenizer, device = load_model()
    encoded = tokenizer(
        article_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(**encoded).logits, dim=-1)[0]
    confidence, predicted_id = probabilities.max(dim=-1)
    classifier_label = ID_TO_LABEL[int(predicted_id.item())]

    model.eval()
    enable_mc_dropout(model)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    fake_probabilities = []
    with torch.no_grad():
        for _ in range(MC_PASSES):
            pass_probabilities = torch.softmax(model(**encoded).logits, dim=-1)[0]
            fake_probabilities.append(pass_probabilities[FAKE_LABEL_ID])

    uncertainty = torch.stack(fake_probabilities).std(unbiased=False).item()
    return classifier_label, confidence.item(), uncertainty


def run_pipeline(article_text: str) -> None:
    """Run every stage while keeping the classifier label as the final decision."""
    classifier_label, confidence, uncertainty = classify_with_uncertainty(article_text)
    verification = verify_claim(article_text)

    verdict = verification.get("verdict", "insufficient")
    max_entailment = float(verification.get("max_entailment", 0.0) or 0.0)
    max_contradiction = float(verification.get("max_contradiction", 0.0) or 0.0)
    evidence_count = len(verification.get("evidence", []))

    load_dotenv()
    try:
        explanation = explain_decision(
            classifier_label=classifier_label,
            confidence=confidence,
            mc_uncertainty=uncertainty,
            nli_verdict=verdict,
            max_entailment=max_entailment,
            max_contradiction=max_contradiction,
            evidence_count=evidence_count,
        )
    except RuntimeError as error:
        if "ANTHROPIC_API_KEY is missing" not in str(error):
            raise
        explanation = None

    print(f"Label: {classifier_label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"MC uncertainty (fake-probability std): {uncertainty:.6f}")
    print(f"NLI verdict (context only): {verdict}")
    if explanation is None:
        print(
            "Explanation: unavailable. Set ANTHROPIC_API_KEY to enable the "
            "explanation step."
        )
    else:
        print("Explanation:")
        print(explanation)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete fake-news decision and explanation pipeline."
    )
    parser.add_argument("text", help="Article text or headline to evaluate")
    args = parser.parse_args()
    run_pipeline(args.text)


if __name__ == "__main__":
    main()
