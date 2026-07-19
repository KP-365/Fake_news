"""Generate a plain-English explanation from structured decision signals only."""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

MODEL_NAME = "claude-3-5-haiku-latest"
MAX_TOKENS = 220
SYSTEM_PROMPT = """You explain an automated fake-news system's completed decision.
The supplied classifier label is final. Do not reclassify it, question it, second-guess it,
or propose a different label. Explain only how the supplied structured signals relate to
that decision. Use plain English, avoid jargon, and write two or three concise sentences.
Do not infer article content, evidence details, or facts that were not supplied."""


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def explain_decision(
    classifier_label: str,
    confidence: float,
    mc_uncertainty: float,
    nli_verdict: str,
    max_entailment: float,
    max_contradiction: float,
    evidence_count: int,
) -> str:
    """Explain a fixed decision using only structured classifier and NLI outputs."""
    if classifier_label not in {"real", "fake"}:
        raise ValueError("classifier_label must be 'real' or 'fake'")
    if nli_verdict not in {"supported", "refuted", "insufficient"}:
        raise ValueError("nli_verdict must be supported, refuted, or insufficient")
    _validate_probability("confidence", confidence)
    if mc_uncertainty < 0:
        raise ValueError("mc_uncertainty must be non-negative")
    _validate_probability("max_entailment", max_entailment)
    _validate_probability("max_contradiction", max_contradiction)
    if not isinstance(evidence_count, int) or evidence_count < 0:
        raise ValueError("evidence_count must be a non-negative integer")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is missing. Set it before requesting an explanation."
        )

    signals = {
        "classifier_label": classifier_label,
        "classifier_confidence": confidence,
        "mc_dropout_uncertainty": mc_uncertainty,
        "nli_verdict": nli_verdict,
        "max_entailment_score": max_entailment,
        "max_contradiction_score": max_contradiction,
        "evidence_count": evidence_count,
    }
    prompt = (
        "Explain why the system reported the fixed classifier label using only these "
        "signals. Treat higher MC Dropout uncertainty as greater model uncertainty. "
        "Mention when the NLI evidence supports, refutes, or cannot resolve the label, "
        "but do not alter or second-guess the label.\n\n"
        f"Structured signals:\n{json.dumps(signals, indent=2)}"
    )

    from anthropic import APIError, Anthropic

    client = Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except APIError as error:
        raise RuntimeError(f"Anthropic explanation request failed: {error}") from error

    explanation = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()
    if not explanation:
        raise RuntimeError("Anthropic returned no explanation text")
    return explanation


def main() -> int:
    load_dotenv()
    try:
        explanation = explain_decision(
            classifier_label="real",
            confidence=0.94,
            mc_uncertainty=0.04,
            nli_verdict="supported",
            max_entailment=0.88,
            max_contradiction=0.03,
            evidence_count=5,
        )
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(explanation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
