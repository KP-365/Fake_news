"""Evidence-based verification tier: retrieve evidence and run an NLI check.

This module is intended for articles gated as low-confidence by MC Dropout. It uses
a pretrained NLI model zero-shot; no verifier weights are trained in this project.
"""

from __future__ import annotations

import multiprocessing
from functools import lru_cache
from multiprocessing.connection import Connection

import spaces

# FEVER-trained NLI model (good for claim verification). Zero-shot.
NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_TOP_K = 5
DEFAULT_VERDICT_THRESHOLD = 0.6  # gamma in the methodology
DDG_HARD_TIMEOUT_SECONDS = 15


@lru_cache(maxsize=1)
def _load_nli(model_name: str = NLI_MODEL_NAME):
    """Load the NLI model once and cache it. Returns (model, tokenizer, device, id2label)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    # Read the model's OWN label order — never hardcode it (varies by checkpoint).
    id2label = {i: label.lower() for i, label in model.config.id2label.items()}
    return model, tokenizer, device, id2label


@spaces.GPU
def nli_scores(premise: str, hypothesis: str, model_name: str = NLI_MODEL_NAME) -> dict:
    """P(entailment / neutral / contradiction) that `premise` supports `hypothesis`."""
    import torch

    model, tokenizer, device, id2label = _load_nli(model_name)
    encoded = tokenizer(
        premise, hypothesis, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**encoded).logits, dim=-1)[0]

    scores = {id2label[i]: float(p) for i, p in enumerate(probs.tolist())}
    return {
        "entailment": scores.get("entailment", 0.0),
        "neutral": scores.get("neutral", 0.0),
        "contradiction": scores.get("contradiction", 0.0),
    }


def _ddg_search_worker(query: str, k: int, sender: Connection) -> None:
    """Run DDG in an isolated process so a stuck network call can be terminated."""
    try:
        try:
            from ddgs import DDGS  # current package name
        except ImportError:
            from duckduckgo_search import DDGS  # older name

        with DDGS(timeout=10) as ddgs:
            hits = list(ddgs.text(query, max_results=k))
        sender.send({"hits": hits})
    except Exception as error:
        sender.send({"error": f"{type(error).__name__}: {error}"})
    finally:
        sender.close()


def _search_ddg_with_hard_timeout(query: str, k: int) -> list[dict]:
    """Return DDG results within a hard wall-clock limit."""
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
    """Retrieve up to k short evidence passages for a claim via web search (no API key).

    WELFake articles have no source URLs, so we search the web by the claim text.
    Swap this for Wikipedia, a news API, or a fixed offline corpus if you want a
    fully reproducible evidence source (recommended for the report).
    """
    query = claim.strip().splitlines()[0][:300]  # headline / first line searches best
    print("Retrieving evidence...", flush=True)

    try:
        hits = _search_ddg_with_hard_timeout(query, k)
    except Exception as error:
        print(f"Evidence retrieval failed: {error}", flush=True)
        hits = []

    evidence = []
    for hit in hits:
        snippet = hit.get("body") or hit.get("snippet") or ""
        if snippet:
            evidence.append({
                "title": hit.get("title", ""),
                "snippet": snippet,
                "url": hit.get("href") or hit.get("url", ""),
            })

    print(f"Retrieved {len(evidence)} evidence result(s).", flush=True)
    return evidence


def verify_claim(
    text: str,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_VERDICT_THRESHOLD,
    model_name: str = NLI_MODEL_NAME,
) -> dict:
    """Retrieve evidence for an article and NLI-check it. Returns a verdict + evidence."""
    # Use the headline / first sentence as the hypothesis — NLI works on claim-sized text.
    claim = text.strip().splitlines()[0][:300] if text.strip() else text[:300]

    evidence = retrieve_evidence(claim, k=top_k)
    if not evidence:
        return {"verdict": "insufficient", "reason": "no evidence retrieved", "evidence": []}

    for item in evidence:
        item["nli"] = nli_scores(item["snippet"], claim, model_name=model_name)

    max_contra = max(e["nli"]["contradiction"] for e in evidence)
    max_entail = max(e["nli"]["entailment"] for e in evidence)

    if max_contra >= threshold and max_contra >= max_entail:
        verdict = "refuted"       # evidence contradicts the claim -> likely fake
    elif max_entail >= threshold:
        verdict = "supported"     # evidence supports the claim -> likely real
    else:
        verdict = "insufficient"  # nothing conclusive -> route to a human

    evidence.sort(
        key=lambda e: max(e["nli"]["contradiction"], e["nli"]["entailment"]),
        reverse=True,
    )
    return {
        "verdict": verdict,
        "max_entailment": max_entail,
        "max_contradiction": max_contra,
        "evidence": evidence,
    }

if __name__ == "__main__":
    import json, sys
    print(json.dumps(verify_claim(sys.argv[1]), indent=2, ensure_ascii=False))