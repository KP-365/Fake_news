"""Hybrid evidence-verification tier for low-confidence fake-news predictions.

Verification order:
1. Search Google's Fact Check Tools API for an existing professional fact-check.
2. When Google has no sufficiently similar, unambiguous match, retrieve web
   snippets with DuckDuckGo and verify them with DeBERTa NLI.

The module does not train verifier weights. It is designed to be called only
for articles selected by the MC Dropout confidence gate.
"""

from __future__ import annotations

import multiprocessing
import os
import re
from difflib import SequenceMatcher
from functools import lru_cache
from multiprocessing.connection import Connection
from typing import Any

import requests

try:
    import spaces
except ImportError:  # Local runs without the ZeroGPU package installed.
    class _SpacesFallback:
        @staticmethod
        def GPU(func):
            return func

    spaces = _SpacesFallback()

NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_TOP_K = 5
DEFAULT_VERDICT_THRESHOLD = 0.6
DDG_HARD_TIMEOUT_SECONDS = 15

GOOGLE_FACTCHECK_ENDPOINT = (
    "https://factchecktools.googleapis.com/v1alpha1/claims:search"
)
GOOGLE_API_KEY_ENV = "GOOGLE_FACTCHECK_API_KEY"
GOOGLE_TIMEOUT_SECONDS = 12
GOOGLE_PAGE_SIZE = 10
GOOGLE_MIN_MATCH_SCORE = 0.45

# Conservative rating normalization. Unclear ratings fall through to DDG + NLI.
_FALSE_PATTERNS = (
    r"\bfalse\b", r"mostly false", r"partly false", r"partially false",
    r"pants on fire", r"four pinocchios", r"three pinocchios",
    r"misleading", r"incorrect", r"inaccurate", r"fabricated", r"fake",
    r"scam", r"hoax", r"altered", r"manipulated", r"no evidence",
)
_TRUE_PATTERNS = (
    r"\btrue\b", r"mostly true", r"largely true", r"correct", r"accurate",
    r"verified", r"supported",
)
_AMBIGUOUS_PATTERNS = (
    r"mixture", r"mixed", r"half true", r"half-true", r"partly true",
    r"unproven", r"unsupported", r"unsubstantiated", r"unclear",
    r"needs context", r"out of context", r"satire", r"legend",
)


def _normalise_text(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text.lower())
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _claim_match_score(query: str, candidate: str) -> float:
    """Return a lightweight lexical similarity score in [0, 1]."""
    query_norm = _normalise_text(query)
    candidate_norm = _normalise_text(candidate)
    if not query_norm or not candidate_norm:
        return 0.0

    sequence_score = SequenceMatcher(None, query_norm, candidate_norm).ratio()
    query_tokens = set(query_norm.split())
    candidate_tokens = set(candidate_norm.split())
    union = query_tokens | candidate_tokens
    jaccard = len(query_tokens & candidate_tokens) / len(union) if union else 0.0
    containment = (
        len(query_tokens & candidate_tokens) / min(len(query_tokens), len(candidate_tokens))
        if query_tokens and candidate_tokens
        else 0.0
    )
    return float(0.45 * sequence_score + 0.35 * jaccard + 0.20 * containment)


def map_google_rating(textual_rating: str) -> str | None:
    """Map publisher-specific ratings to supported/refuted, or None if unclear."""
    rating = _normalise_text(textual_rating)
    if not rating:
        return None
    if any(re.search(pattern, rating) for pattern in _AMBIGUOUS_PATTERNS):
        return None
    if any(re.search(pattern, rating) for pattern in _FALSE_PATTERNS):
        return "refuted"
    if any(re.search(pattern, rating) for pattern in _TRUE_PATTERNS):
        return "supported"
    return None


def search_google_factchecks(
    claim: str,
    api_key: str | None = None,
    page_size: int = GOOGLE_PAGE_SIZE,
    timeout: int = GOOGLE_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Search Google Fact Check Tools and return normalized review candidates.

    If no API key is configured, an empty list is returned so the pipeline can
    continue through DDG + DeBERTa without failing.
    """
    key = api_key or os.getenv(GOOGLE_API_KEY_ENV, "").strip()
    if not key:
        return []

    query = claim.strip().splitlines()[0][:300]
    response = requests.get(
        GOOGLE_FACTCHECK_ENDPOINT,
        params={
            "query": query,
            "languageCode": "en",
            "pageSize": page_size,
            "key": key,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    candidates: list[dict[str, Any]] = []
    for matched_claim in payload.get("claims", []):
        matched_text = matched_claim.get("text", "")
        match_score = _claim_match_score(query, matched_text)
        for review in matched_claim.get("claimReview", []):
            publisher = review.get("publisher") or {}
            rating = review.get("textualRating", "")
            candidates.append(
                {
                    "matched_claim": matched_text,
                    "claimant": matched_claim.get("claimant", ""),
                    "claim_date": matched_claim.get("claimDate", ""),
                    "publisher": publisher.get("name", ""),
                    "publisher_site": publisher.get("site", ""),
                    "review_title": review.get("title", ""),
                    "review_url": review.get("url", ""),
                    "review_date": review.get("reviewDate", ""),
                    "textual_rating": rating,
                    "mapped_verdict": map_google_rating(rating),
                    "match_score": match_score,
                }
            )

    candidates.sort(key=lambda item: item["match_score"], reverse=True)
    return candidates


def verify_with_google_factcheck(
    claim: str,
    min_match_score: float = GOOGLE_MIN_MATCH_SCORE,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Return a usable Google verdict or an insufficient result with metadata."""
    try:
        candidates = search_google_factchecks(claim, api_key=api_key)
    except Exception as error:
        return {
            "verdict": "insufficient",
            "source": "google_fact_check",
            "reason": f"Google Fact Check request failed: {type(error).__name__}: {error}",
            "google_available": bool(api_key or os.getenv(GOOGLE_API_KEY_ENV, "").strip()),
            "google_candidates": [],
        }

    if not candidates:
        reason = (
            f"{GOOGLE_API_KEY_ENV} is not configured"
            if not (api_key or os.getenv(GOOGLE_API_KEY_ENV, "").strip())
            else "no Google fact-check match"
        )
        return {
            "verdict": "insufficient",
            "source": "google_fact_check",
            "reason": reason,
            "google_available": bool(api_key or os.getenv(GOOGLE_API_KEY_ENV, "").strip()),
            "google_candidates": [],
        }

    usable = [
        item
        for item in candidates
        if item["match_score"] >= min_match_score and item["mapped_verdict"] is not None
    ]
    if not usable:
        return {
            "verdict": "insufficient",
            "source": "google_fact_check",
            "reason": "matches were weak or ratings were ambiguous",
            "google_available": True,
            "google_candidates": candidates,
        }

    # Use the strongest match. If equally strong reviews conflict, defer to DDG/NLI.
    best_score = usable[0]["match_score"]
    strongest = [item for item in usable if abs(item["match_score"] - best_score) < 0.02]
    strongest_verdicts = {item["mapped_verdict"] for item in strongest}
    if len(strongest_verdicts) > 1:
        return {
            "verdict": "insufficient",
            "source": "google_fact_check",
            "reason": "conflicting high-similarity Google ratings",
            "google_available": True,
            "google_candidates": candidates,
        }

    best = usable[0]
    return {
        "verdict": best["mapped_verdict"],
        "source": "google_fact_check",
        "reason": "usable professional fact-check found",
        "google_available": True,
        "google_match": best,
        "google_candidates": candidates,
    }


@lru_cache(maxsize=1)
def _load_nli(model_name: str = NLI_MODEL_NAME):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    id2label = {i: label.lower() for i, label in model.config.id2label.items()}
    return model, tokenizer, device, id2label


@spaces.GPU
def nli_scores(premise: str, hypothesis: str, model_name: str = NLI_MODEL_NAME) -> dict:
    import torch

    model, tokenizer, device, id2label = _load_nli(model_name)
    encoded = tokenizer(
        premise, hypothesis, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(**encoded).logits, dim=-1)[0]

    scores = {id2label[i]: float(value) for i, value in enumerate(probabilities.tolist())}
    return {
        "entailment": scores.get("entailment", 0.0),
        "neutral": scores.get("neutral", 0.0),
        "contradiction": scores.get("contradiction", 0.0),
    }


def _ddg_search_worker(query: str, k: int, sender: Connection) -> None:
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS(timeout=10) as ddgs:
            hits = list(ddgs.text(query, max_results=k))
        sender.send({"hits": hits})
    except Exception as error:
        sender.send({"error": f"{type(error).__name__}: {error}"})
    finally:
        sender.close()


def _search_ddg_with_hard_timeout(query: str, k: int) -> list[dict]:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_ddg_search_worker, args=(query, k, sender), daemon=True
    )
    process.start()
    sender.close()

    try:
        if not receiver.poll(DDG_HARD_TIMEOUT_SECONDS):
            raise TimeoutError(
                f"DDG retrieval exceeded {DDG_HARD_TIMEOUT_SECONDS} seconds"
            )
        message = receiver.recv()
    finally:
        receiver.close()
        process.join(timeout=0.1)
        if process.is_alive():
            process.terminate()
            process.join()

    if "error" in message:
        raise RuntimeError(message["error"])
    return message["hits"]


def retrieve_evidence(claim: str, k: int = DEFAULT_TOP_K) -> list[dict]:
    query = claim.strip().splitlines()[0][:300]
    print("Retrieving DDG evidence...", flush=True)
    try:
        hits = _search_ddg_with_hard_timeout(query, k)
    except Exception as error:
        print(f"DDG retrieval failed: {error}", flush=True)
        hits = []

    evidence = []
    for hit in hits:
        snippet = hit.get("body") or hit.get("snippet") or ""
        if snippet:
            evidence.append(
                {
                    "title": hit.get("title", ""),
                    "snippet": snippet,
                    "url": hit.get("href") or hit.get("url", ""),
                }
            )
    print(f"Retrieved {len(evidence)} DDG evidence result(s).", flush=True)
    return evidence


def verify_with_ddg_nli(
    claim: str,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_VERDICT_THRESHOLD,
    model_name: str = NLI_MODEL_NAME,
) -> dict[str, Any]:
    evidence = retrieve_evidence(claim, k=top_k)
    if not evidence:
        return {
            "verdict": "insufficient",
            "source": "none",
            "reason": "no DDG evidence retrieved",
            "evidence": [],
        }

    for item in evidence:
        item["nli"] = nli_scores(item["snippet"], claim, model_name=model_name)

    max_contradiction = max(item["nli"]["contradiction"] for item in evidence)
    max_entailment = max(item["nli"]["entailment"] for item in evidence)

    if max_contradiction >= threshold and max_contradiction >= max_entailment:
        verdict = "refuted"
    elif max_entailment >= threshold:
        verdict = "supported"
    else:
        verdict = "insufficient"

    evidence.sort(
        key=lambda item: max(
            item["nli"]["contradiction"], item["nli"]["entailment"]
        ),
        reverse=True,
    )
    return {
        "verdict": verdict,
        "source": "ddg_deberta" if verdict != "insufficient" else "none",
        "reason": "DDG evidence evaluated with DeBERTa NLI",
        "max_entailment": max_entailment,
        "max_contradiction": max_contradiction,
        "evidence": evidence,
    }


def verify_claim(
    text: str,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_VERDICT_THRESHOLD,
    model_name: str = NLI_MODEL_NAME,
    google_min_match_score: float = GOOGLE_MIN_MATCH_SCORE,
    google_api_key: str | None = None,
) -> dict[str, Any]:
    """Verify a claim using Google first and DDG + DeBERTa as fallback."""
    claim = text.strip().splitlines()[0][:300] if text.strip() else text[:300]

    google_result = verify_with_google_factcheck(
        claim,
        min_match_score=google_min_match_score,
        api_key=google_api_key,
    )
    if google_result["verdict"] in {"supported", "refuted"}:
        return google_result

    ddg_result = verify_with_ddg_nli(
        claim,
        top_k=top_k,
        threshold=threshold,
        model_name=model_name,
    )
    ddg_result["google_reason"] = google_result.get("reason", "")
    ddg_result["google_available"] = google_result.get("google_available", False)
    ddg_result["google_candidates"] = google_result.get("google_candidates", [])
    return ddg_result


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(verify_claim(sys.argv[1]), indent=2, ensure_ascii=False))
