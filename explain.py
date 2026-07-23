"""Generate a plain-English explanation from structured decision signals only."""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

MODEL_NAME = "claude-haiku-4-5"
MAX_TOKENS = 220
SYSTEM_PROMPT = """You explain an automated fake-news system's completed decision.
The supplied classifier label is final. Do not reclassify it, question it, second-guess it,
or imply that another label may be more accurate. Explain only how the supplied structured
signals relate to that fixed decision. Every technical signal must be introduced by its
plain-English meaning, immediately followed by its technical name in square brackets.
Always cover these three signals with the plain-English wording first: how sure the classifier
was [confidence], how consistent the classifier was across repeated runs [Monte Carlo dropout
uncertainty], and how well the retrieved evidence agreed with the claim [NLI]. If NLI conflicts
with the label, describe the conflict as context and explicitly state that it does not change
the classifier label. Avoid unexplained jargon and em dashes, and write two or three concise
sentences. Do not infer article content, evidence details, or facts that were not supplied."""


class MissingAnthropicKeyError(RuntimeError):
    """Raised when neither a per-request nor environment API key is available."""


class AnthropicKeyRejectedError(RuntimeError):
    """Raised when Anthropic rejects the supplied API key."""


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
    api_key: str | None = None,
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

    resolved_api_key = (
        os.getenv("ANTHROPIC_API_KEY", "").strip()
        if api_key is None
        else api_key.strip()
    )
    if not resolved_api_key:
        raise MissingAnthropicKeyError(
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
        "signals. Cover how sure the classifier was [confidence], how consistent the "
        "classifier was across repeated runs [Monte Carlo dropout uncertainty], and how "
        "well the retrieved evidence agreed with the claim [NLI]. Treat higher Monte Carlo "
        "dropout uncertainty as less repeated-run stability. Mention when NLI evidence "
        "supports, refutes, or cannot resolve the label. A conflicting NLI verdict is "
        "context only: state that it does not change the classifier label, and do not imply "
        "the label is wrong.\n\n"
        f"Structured signals:\n{json.dumps(signals, indent=2)}"
    )

    from anthropic import APIError, Anthropic, AuthenticationError

    client = Anthropic(api_key=resolved_api_key)
    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except AuthenticationError:
        raise AnthropicKeyRejectedError("Anthropic API key was rejected") from None
    except APIError:
        raise RuntimeError("Anthropic explanation request failed") from None

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
