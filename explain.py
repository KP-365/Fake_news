"""Generate a plain-English explanation from structured decision signals only."""

from __future__ import annotations

import json
import os
import re
import sys

from dotenv import load_dotenv

MODEL_NAME = "claude-haiku-4-5"
MAX_TOKENS = 220
SYSTEM_PROMPT = """You explain an automated fake-news system's completed decision.
The supplied label is final. Do not reclassify it, question it, second-guess it, or imply that
another label may be more accurate. Explain only how the supplied structured signals relate to
that fixed decision. Write exactly one concise paragraph with no bullets or line breaks. State
every supplied display value and immediately explain it in plain English. In the user-facing
paragraph, never use these technical terms: dropout, standard deviation, NLI, entailment, or
contradiction. Describe confidence as the fixed system run's strength of preference for the label,
not the chance that the label is correct. Describe repeated-check variation as consistency:
smaller variation means more consistent results. Describe evidence scores as supporting and
conflicting matches, not as truth
probabilities or source-quality measures. Conflicting evidence is context only and does not change
the final label. If evidence is insufficient, say that the evidence checks did not resolve the
claim. Do not infer article content, evidence details, or facts that were not supplied."""

FORBIDDEN_OUTPUT_TERMS = (
    "dropout",
    "standard deviation",
    "nli",
    "entailment",
    "contradiction",
)


class MissingAnthropicKeyError(RuntimeError):
    """Raised when neither a per-request nor environment API key is available."""


class AnthropicKeyRejectedError(RuntimeError):
    """Raised when Anthropic rejects the supplied API key."""


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _required_display_values(
    confidence: float,
    mc_uncertainty: float,
    max_entailment: float,
    max_contradiction: float,
) -> tuple[str, str, str, str]:
    """Return the exact user-facing values derived from the internal metrics."""
    return (
        f"{confidence:.1%}",
        f"{mc_uncertainty * 100:.1f} percentage points",
        f"{max_entailment:.1%}",
        f"{max_contradiction:.1%}",
    )


def _display_guidance(
    classifier_label: str,
    confidence: float,
    mc_uncertainty: float,
    nli_verdict: str,
    max_entailment: float,
    max_contradiction: float,
    evidence_count: int,
) -> str:
    """Build plain-language display guidance while retaining internal metric inputs."""
    confidence_display, variation_display, support_display, conflict_display = (
        _required_display_values(
            confidence, mc_uncertainty, max_entailment, max_contradiction
        )
    )
    if nli_verdict == "insufficient":
        verdict_guidance = (
            "The evidence verdict was insufficient: the evidence checks did not resolve the "
            "claim. Do not treat low or zero scores as affirmative evidence."
        )
    else:
        verdict_meaning = (
            "at least one retrieved passage supported the claim"
            if nli_verdict == "supported"
            else "at least one retrieved passage conflicted with the claim"
        )
        verdict_guidance = (
            f"The evidence verdict was {nli_verdict}, meaning {verdict_meaning}."
        )
        conflicts_with_label = (
            nli_verdict == "supported" and classifier_label == "fake"
        ) or (nli_verdict == "refuted" and classifier_label == "real")
        if conflicts_with_label:
            verdict_guidance += (
                " This conflicts with the fixed label; state immediately that it is context "
                "only and does not change that label."
            )

    return (
        "Use these exact user-facing values and meanings in the paragraph:\n"
        f"- {confidence_display} confidence: the fixed system run's strength of preference "
        "for the selected label; it is not the chance that the label is correct.\n"
        f"- {variation_display} repeated-check variation: this shows how much the fake score "
        "varied across 30 checks; smaller variation means more consistent results.\n"
        f"- {support_display} strongest supporting match: the retrieved passage that most "
        "strongly supported the claim.\n"
        f"- {conflict_display} strongest conflicting match: the retrieved passage that most "
        "strongly conflicted with the claim.\n"
        "The two evidence percentages are match scores, not truth probabilities or "
        f"source-quality ratings. {verdict_guidance} Evidence passages retrieved: "
        f"{evidence_count}."
    )


def _validate_explanation_output(
    explanation: str,
    *,
    confidence: float,
    mc_uncertainty: float,
    max_entailment: float,
    max_contradiction: float,
) -> str:
    """Require one safe paragraph containing every derived display value."""
    normalized_explanation = explanation.strip()
    if not normalized_explanation or len(normalized_explanation.splitlines()) != 1:
        raise RuntimeError("Anthropic explanation must be exactly one paragraph")

    required_values = _required_display_values(
        confidence, mc_uncertainty, max_entailment, max_contradiction
    )
    missing_values = [
        value for value in required_values if value not in normalized_explanation
    ]
    if missing_values:
        raise RuntimeError(
            "Anthropic explanation omitted required display value(s): "
            + ", ".join(missing_values)
        )

    forbidden_terms = [
        term
        for term in FORBIDDEN_OUTPUT_TERMS
        if re.search(rf"\b{re.escape(term)}\b", normalized_explanation, re.IGNORECASE)
    ]
    if forbidden_terms:
        raise RuntimeError(
            "Anthropic explanation used forbidden technical term(s): "
            + ", ".join(forbidden_terms)
        )
    return normalized_explanation


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
    display_guidance = _display_guidance(
        classifier_label,
        confidence,
        mc_uncertainty,
        nli_verdict,
        max_entailment,
        max_contradiction,
        evidence_count,
    )
    prompt = (
        "Explain the fixed classifier label in exactly one paragraph using only the "
        "structured signals and the required user-facing wording below. Put each number's "
        "plain-English interpretation immediately after that number. Do not omit or rename "
        "a metric.\n\n"
        f"{display_guidance}\n\n"
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
    return _validate_explanation_output(
        explanation,
        confidence=confidence,
        mc_uncertainty=mc_uncertainty,
        max_entailment=max_entailment,
        max_contradiction=max_contradiction,
    )


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
