"""Run inference with the fine-tuned RoBERTa LoRA checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

CHECKPOINT_DIR = Path(__file__).resolve().parent / "models" / "roberta-trained-welfake"
BASE_MODEL_NAME = "roberta-base"
MAX_LENGTH = 256
ID_TO_LABEL = {0: "real", 1: "fake"}
# Upper bounds translate fake-probability variation into one shared stability label.
MC_STABILITY_THRESHOLDS = (
    (0.02, "Very stable"),
    (0.05, "Stable"),
    (0.10, "Somewhat unstable"),
    (float("inf"), "Unstable"),
)


def describe_mc_stability(uncertainty: float) -> str:
    """Return the plain-language label for an MC Dropout uncertainty value."""
    return next(
        label for upper_bound, label in MC_STABILITY_THRESHOLDS
        if uncertainty < upper_bound
    )


def enable_mc_dropout(model: Any) -> None:
    """Enable only dropout layers while the rest of the model remains in eval mode."""
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def _select_startup_device(torch_module: Any) -> Any:
    """Keep ZeroGPU startup on CPU; otherwise select ordinary CUDA/CPU."""
    is_zero_gpu = os.getenv("SPACES_ZERO_GPU", "").strip().lower() in {
        "1",
        "t",
        "true",
    }
    if is_zero_gpu:
        return torch_module.device("cpu")
    return torch_module.device(
        "cuda" if torch_module.cuda.is_available() else "cpu"
    )


def load_model(checkpoint_dir: Path = CHECKPOINT_DIR) -> tuple[Any, Any, Any]:
    """Load the tokenizer, base model, and saved PEFT adapter."""
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_dir}. "
            "Finish training the notebook first."
        )

    classifier_head_path = checkpoint_dir / "classifier_head.pt"
    if not classifier_head_path.is_file():
        raise FileNotFoundError(
            f"Classifier head not found at {classifier_head_path}. "
            "Save classifier.state_dict() there before running prediction."
        )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = _select_startup_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id={label: index for index, label in ID_TO_LABEL.items()},
    )
    model = PeftModel.from_pretrained(
        base_model, checkpoint_dir, torch_device="cpu"
    )
    classifier_state = torch.load(
        classifier_head_path, map_location="cpu", weights_only=True
    )
    base_model.classifier.load_state_dict(classifier_state)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict(text: str, checkpoint_dir: Path = CHECKPOINT_DIR) -> tuple[str, float]:
    """Predict a fake/real label and confidence for one text string."""
    model, tokenizer, device = load_model(checkpoint_dir)
    import torch
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

    with torch.no_grad():
        probabilities = torch.softmax(model(**encoded).logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities).item())
    return ID_TO_LABEL[predicted_id], float(probabilities[predicted_id].item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify text with the fine-tuned RoBERTa fake-news model."
    )
    parser.add_argument("text", help="Text to classify")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        label, confidence = predict(args.text)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"{label} (confidence: {confidence:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
