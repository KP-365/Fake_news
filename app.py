"""Local Gradio interface for the fake-news analysis pipeline."""

from __future__ import annotations

import html
from multiprocessing import current_process
from threading import Lock
from typing import Any
from urllib.parse import urlparse

import gradio as gr

from explain import (
    AnthropicKeyRejectedError,
    MissingAnthropicKeyError,
    explain_decision,
)
from pipeline import classify_with_uncertainty
from predict import describe_mc_stability, load_model
from verify import verify_claim

DEFAULT_ARTICLE = "Federal Reserve holds interest rates steady amid mixed economic data"
MISSING_KEY_MESSAGE = (
    "Explanation unavailable. Add an Anthropic API key above to enable the Claude "
    "explanation step."
)
REJECTED_KEY_MESSAGE = (
    "Explanation unavailable: key rejected. Check the Anthropic API key and try again."
)

CLASSIFIER_MODEL: Any | None = None
CLASSIFIER_TOKENIZER: Any | None = None
CLASSIFIER_DEVICE: Any | None = None
CLASSIFIER_LOCK = Lock()


def initialize_classifier() -> None:
    """Load one shared classifier before the local server starts."""
    global CLASSIFIER_MODEL, CLASSIFIER_TOKENIZER, CLASSIFIER_DEVICE
    if CLASSIFIER_MODEL is None:
        CLASSIFIER_MODEL, CLASSIFIER_TOKENIZER, CLASSIFIER_DEVICE = load_model()


def _safe_evidence_url(url: object) -> str | None:
    """Return only HTTP(S) evidence URLs."""
    if not isinstance(url, str):
        return None
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return html.escape(url, quote=True)


def _decision_html(
    classifier_label: str, confidence: float, uncertainty: float, verdict: str
) -> str:
    label_class = "label-real" if classifier_label == "real" else "label-fake"
    stability = describe_mc_stability(uncertainty)
    return f"""
    <section class="decision-record" aria-label="Analysis result">
      <div class="decision-primary">
        <span class="signal-label">Final classifier label</span>
        <div class="label-line">
          <strong class="decision-label {label_class}">{html.escape(classifier_label.upper())}</strong>
          <span class="confidence-value">{confidence:.2%} confidence</span>
        </div>
        <p>The classifier label is final. Verification is supporting context only.</p>
      </div>
      <dl class="signal-list">
        <div>
          <dt>Prediction stability</dt>
          <dd class="stability-value">{html.escape(stability)}</dd>
          <span>MC Dropout uncertainty (fake-probability std): {uncertainty:.6f}</span>
        </div>
        <div>
          <dt>NLI verdict</dt>
          <dd>{html.escape(verdict.capitalize())}</dd>
          <span>Context only, never an override</span>
        </div>
      </dl>
    </section>
    """


def _evidence_html(evidence: list[dict], retrieval_note: str | None = None) -> str:
    if retrieval_note:
        evidence_body = f'<p class="evidence-empty">{html.escape(retrieval_note)}</p>'
    elif not evidence:
        evidence_body = (
            '<p class="evidence-empty">No retrieved evidence was available for this claim.</p>'
        )
    else:
        items = []
        for position, item in enumerate(evidence, start=1):
            title = html.escape(str(item.get("title") or f"Evidence source {position}"))
            snippet = html.escape(str(item.get("snippet") or "No snippet available."))
            safe_url = _safe_evidence_url(item.get("url"))
            title_markup = (
                f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer">{title}</a>'
                if safe_url
                else f"<strong>{title}</strong>"
            )
            nli = item.get("nli") or {}
            entailment = float(nli.get("entailment", 0.0) or 0.0)
            contradiction = float(nli.get("contradiction", 0.0) or 0.0)
            items.append(
                f"""
                <li>
                  <div class="evidence-title">{title_markup}</div>
                  <p>{snippet}</p>
                  <div class="evidence-scores">Entailment {entailment:.3f} · Contradiction {contradiction:.3f}</div>
                </li>
                """
            )
        evidence_body = f'<ol class="evidence-list">{"".join(items)}</ol>'

    return f"""
    <section class="evidence-section" aria-label="Retrieved evidence context">
      <div class="section-heading">
        <div>
          <span class="signal-label">Verification layer</span>
          <h2>Retrieved evidence</h2>
        </div>
        <span class="context-marker">Context only</span>
      </div>
      <p class="context-note">These sources inform the NLI verdict. They never change the classifier label.</p>
      {evidence_body}
    </section>
    """


def analyze_article(article_text: str, api_key: str) -> tuple[str, str, str, str]:
    """Run the full pipeline with one shared classifier and an ephemeral API key."""
    article_text = article_text.strip()
    if not article_text:
        raise gr.Error("Enter a headline or article before running the analysis.")

    if any(
        component is None
        for component in (CLASSIFIER_MODEL, CLASSIFIER_TOKENIZER, CLASSIFIER_DEVICE)
    ):
        raise RuntimeError("Classifier is not initialized")

    with CLASSIFIER_LOCK:
        classifier_label, confidence, uncertainty = classify_with_uncertainty(
            article_text,
            model=CLASSIFIER_MODEL,
            tokenizer=CLASSIFIER_TOKENIZER,
            device=CLASSIFIER_DEVICE,
        )

    retrieval_note = None
    try:
        verification = verify_claim(article_text)
    except Exception as error:
        verification = {"verdict": "insufficient", "evidence": []}
        retrieval_note = f"Verification was unavailable: {error}"

    verdict = str(verification.get("verdict", "insufficient"))
    if verdict not in {"supported", "refuted", "insufficient"}:
        verdict = "insufficient"
    evidence = verification.get("evidence") or []
    max_entailment = float(verification.get("max_entailment", 0.0) or 0.0)
    max_contradiction = float(verification.get("max_contradiction", 0.0) or 0.0)

    try:
        explanation = explain_decision(
            classifier_label=classifier_label,
            confidence=confidence,
            mc_uncertainty=uncertainty,
            nli_verdict=verdict,
            max_entailment=max_entailment,
            max_contradiction=max_contradiction,
            evidence_count=len(evidence),
            api_key=api_key.strip(),
        )
        explanation_markdown = (
            "### Claude explanation\n\n"
            f"{explanation.replace(chr(0x2014), ', ')}\n\n"
            "*Generated from structured scores only. No article or evidence text was sent to Claude.*"
        )
    except MissingAnthropicKeyError:
        explanation_markdown = f"### Claude explanation\n\n{MISSING_KEY_MESSAGE}"
    except AnthropicKeyRejectedError:
        explanation_markdown = f"### Claude explanation\n\n{REJECTED_KEY_MESSAGE}"
    except RuntimeError:
        explanation_markdown = (
            "### Claude explanation\n\n"
            "Explanation unavailable. The Claude service could not complete this request."
        )

    return (
        _decision_html(classifier_label, confidence, uncertainty, verdict),
        _evidence_html(evidence, retrieval_note),
        explanation_markdown,
        "",
    )


INITIAL_DECISION = """
<section class="decision-record decision-empty" aria-label="Waiting for analysis">
  <span class="signal-label">Analysis result</span>
  <h2>Ready for an article</h2>
  <p>Submit text to see the fixed classifier label and uncertainty signals.</p>
</section>
"""
INITIAL_EVIDENCE = """
<section class="evidence-section evidence-waiting" aria-label="Evidence waiting state">
  <span class="signal-label">Verification layer</span>
  <h2>Evidence will appear here</h2>
  <p>Retrieved links are always shown as context, never as a label override.</p>
</section>
"""
INITIAL_EXPLANATION = """### Claude explanation

The numbers-only explanation will appear after analysis.
"""


def build_app() -> gr.Blocks:
    """Build the Gradio interface without reloading model weights."""
    with gr.Blocks(
        title="News Signal Review", fill_width=False, analytics_enabled=False
    ) as demo:
        gr.HTML(
            """
            <header class="app-header">
              <div>
                <span class="process-label">CLASSIFY / CHECK / EXPLAIN</span>
                <h1>News Signal Review</h1>
                <p>Inspect a fixed RoBERTa label alongside uncertainty, web evidence, and a numbers-only explanation.</p>
              </div>
              <div class="model-note">
                <strong>Decision policy</strong>
                <span>Classifier label stays final</span>
              </div>
            </header>
            """,
            elem_classes="header-block",
        )

        with gr.Row(equal_height=False, elem_classes="workspace"):
            with gr.Column(scale=5, min_width=340, elem_classes="input-panel"):
                gr.Markdown(
                    "## Article input\nPaste a headline or article body. Longer text is truncated to the model's 256-token window."
                )
                article_input = gr.Textbox(
                    value=DEFAULT_ARTICLE,
                    label="Article text",
                    info="The text is used locally by RoBERTa and for web evidence retrieval.",
                    lines=9,
                    max_lines=16,
                    placeholder="Paste a headline or article body",
                    elem_classes="article-input",
                )
                with gr.Accordion(
                    "Optional: Claude explanation key",
                    open=False,
                    elem_classes="api-key-accordion",
                ):
                    gr.Markdown(
                        "Use your own Anthropic key for this request. It is cleared after analysis, never stored, and never added to the environment."
                    )
                    api_key_input = gr.Textbox(
                        label="Anthropic API key",
                        type="password",
                        placeholder="sk-ant-...",
                        info="Leave empty to run classification and verification without Claude.",
                        elem_classes="api-key-input",
                        preserved_by_key=[],
                    )
                analyze_button = gr.Button(
                    "Analyze article",
                    variant="primary",
                    size="lg",
                    elem_classes="primary-action",
                )
                gr.Markdown(
                    "**Processing note:** the first verification run downloads the NLI model. Analysis can take 20 to 60 seconds.",
                    elem_classes="processing-note",
                )

            with gr.Column(scale=7, min_width=420, elem_classes="results-panel"):
                decision_output = gr.HTML(INITIAL_DECISION, elem_classes="decision-output")
                evidence_output = gr.HTML(INITIAL_EVIDENCE, elem_classes="evidence-output")
                explanation_output = gr.Markdown(
                    INITIAL_EXPLANATION, elem_classes="explanation-output"
                )

        inputs = [article_input, api_key_input]
        outputs = [decision_output, evidence_output, explanation_output, api_key_input]
        analyze_button.click(
            fn=analyze_article,
            inputs=inputs,
            outputs=outputs,
            api_name="analyze",
            api_visibility="public",
        )
        article_input.submit(
            fn=analyze_article,
            inputs=inputs,
            outputs=outputs,
            api_visibility="private",
        )
        api_key_input.submit(
            fn=analyze_article,
            inputs=inputs,
            outputs=outputs,
            api_visibility="private",
        )

    return demo.queue(max_size=8, default_concurrency_limit=1)


APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500&family=Manrope:wght@400;500;600;700&display=swap');

:root {
  --paper: #f4f1ea;
  --surface: #fffdf8;
  --ink: #18212b;
  --muted: #52606d;
  --line: #c9c4b8;
  --line-strong: #8b918f;
  --cobalt: #174ea6;
  --cobalt-dark: #103b7d;
  --real: #17663a;
  --fake: #a12b2b;
  --context: #80520c;
  --focus: #f2a900;
}

body,
.gradio-container {
  background: var(--paper) !important;
  color: var(--ink) !important;
  font-family: 'Manrope', ui-sans-serif, sans-serif !important;
}

.gradio-container {
  max-width: 1240px !important;
  padding: 32px 28px 64px !important;
}

.app-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 32px;
  padding: 8px 0 28px;
  border-bottom: 2px solid var(--ink);
}

.app-header h1 {
  margin: 8px 0 10px;
  color: var(--ink);
  font-size: clamp(2.25rem, 5vw, 4.5rem);
  font-weight: 600;
  letter-spacing: -0.055em;
  line-height: 0.98;
}

.app-header p {
  max-width: 720px;
  margin: 0;
  color: var(--muted);
  font-size: 1.02rem;
  line-height: 1.65;
}

.process-label,
.signal-label {
  color: var(--cobalt-dark);
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}

.model-note {
  flex: 0 0 auto;
  min-width: 220px;
  padding-left: 18px;
  border-left: 3px solid var(--context);
}

.model-note strong,
.model-note span {
  display: block;
}

.model-note strong {
  margin-bottom: 4px;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.model-note span {
  color: var(--muted);
  font-size: 0.9rem;
}

.workspace {
  gap: 28px !important;
  margin-top: 28px;
  align-items: flex-start;
}

.input-panel,
.results-panel {
  gap: 16px !important;
}

.input-panel {
  position: sticky;
  top: 24px;
  padding: 24px !important;
  border: 1px solid var(--line) !important;
  border-radius: 16px !important;
  background: var(--surface) !important;
  box-shadow: 0 12px 32px rgba(24, 33, 43, 0.06);
}

.input-panel h2 {
  margin: 0 0 6px;
  color: var(--ink);
  font-size: 1.35rem;
}

.input-panel p,
.processing-note {
  color: var(--muted) !important;
  line-height: 1.55;
}

.article-input textarea {
  min-height: 220px !important;
  color: var(--ink) !important;
  background: #ffffff !important;
  border-color: var(--line-strong) !important;
  line-height: 1.55 !important;
}

.article-input .info,
.api-key-input .info {
  color: var(--muted) !important;
  opacity: 1 !important;
}

.api-key-accordion {
  overflow: hidden;
  border: 1px solid var(--line) !important;
  border-radius: 10px !important;
  background: #faf8f2 !important;
}

.api-key-accordion p {
  color: var(--muted) !important;
  font-size: 0.82rem;
  line-height: 1.5;
}

.api-key-input input {
  color: var(--ink) !important;
  background: #ffffff !important;
}

.article-input textarea::placeholder,
.api-key-input input::placeholder {
  color: #5f6b76 !important;
  opacity: 1 !important;
}

.article-input textarea:focus-visible,
.api-key-input input:focus-visible,
.primary-action:focus-visible,
a:focus-visible {
  outline: 3px solid var(--focus) !important;
  outline-offset: 3px !important;
}

.primary-action {
  min-height: 48px !important;
  border: 1px solid var(--cobalt) !important;
  border-radius: 10px !important;
  background: var(--cobalt) !important;
  color: white !important;
  font-weight: 700 !important;
  transition: background-color 160ms ease, border-color 160ms ease, box-shadow 160ms ease !important;
}

.primary-action:hover {
  border-color: var(--cobalt-dark) !important;
  background: var(--cobalt-dark) !important;
  box-shadow: 0 4px 14px rgba(23, 78, 166, 0.2) !important;
}

.decision-output,
.evidence-output,
.explanation-output {
  overflow: visible;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

.decision-record,
.evidence-section,
.explanation-output > div {
  overflow: hidden;
  padding: 24px;
  border: 1px solid var(--line);
  border-radius: 16px;
  background: var(--surface);
}

.decision-primary {
  padding-bottom: 22px;
  border-bottom: 1px solid var(--line);
}

.label-line {
  display: flex;
  align-items: baseline;
  gap: 14px;
  margin: 8px 0;
}

.decision-label {
  font-size: clamp(2.2rem, 6vw, 4rem);
  font-weight: 700;
  letter-spacing: -0.045em;
  line-height: 1;
}

.label-real { color: var(--real); }
.label-fake { color: var(--fake); }

.confidence-value,
.signal-list dd,
.evidence-scores {
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
}

.confidence-value {
  color: var(--muted);
  font-size: 0.9rem;
}

.decision-primary p,
.context-note,
.evidence-list p,
.evidence-empty,
.evidence-waiting p,
.decision-empty p {
  color: var(--muted);
  line-height: 1.6;
}

.signal-list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0;
  margin: 0;
}

.signal-list > div {
  padding: 20px 20px 0 0;
}

.signal-list > div + div {
  padding-left: 20px;
  border-left: 1px solid var(--line);
}

.signal-list dt {
  color: var(--muted);
  font-size: 0.78rem;
  font-weight: 600;
}

.signal-list dd {
  margin: 6px 0;
  color: var(--ink);
  font-size: 1.25rem;
  font-weight: 500;
}

.signal-list .stability-value {
  color: var(--cobalt-dark);
  font-size: 1.35rem;
  font-weight: 700;
}

.signal-list span {
  display: block;
  color: var(--muted);
  font-size: 0.78rem;
  line-height: 1.45;
}

.section-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
}

.section-heading h2,
.evidence-waiting h2,
.decision-empty h2 {
  margin: 5px 0 0;
  color: var(--ink);
  font-size: 1.35rem;
}

.context-marker {
  flex: 0 0 auto;
  padding: 6px 9px;
  border: 1px solid #b48538;
  border-radius: 4px;
  color: var(--context);
  background: #fff5dc;
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
  font-size: 0.7rem;
  font-weight: 500;
  text-transform: uppercase;
}

.context-note {
  margin: 12px 0 18px;
  padding: 12px 14px;
  border-left: 3px solid #b48538;
  background: #fff8e8;
}

.evidence-list {
  margin: 0;
  padding: 0;
  list-style: none;
  counter-reset: evidence;
}

.evidence-list li {
  position: relative;
  padding: 18px 0 18px 38px;
  border-top: 1px solid var(--line);
  counter-increment: evidence;
}

.evidence-list li::before {
  position: absolute;
  top: 18px;
  left: 0;
  color: var(--muted);
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
  font-size: 0.75rem;
  content: counter(evidence, decimal-leading-zero);
}

.evidence-title a,
.evidence-title strong {
  color: var(--cobalt-dark);
  font-weight: 700;
  text-decoration-thickness: 1px;
  text-underline-offset: 3px;
}

.evidence-title a:hover {
  color: var(--cobalt);
  text-decoration-thickness: 2px;
}

.evidence-list p {
  margin: 7px 0;
  font-size: 0.9rem;
}

.evidence-scores {
  color: var(--muted);
  font-size: 0.72rem;
}

.explanation-output h3 {
  margin-top: 0;
  color: var(--ink);
  font-size: 1.35rem;
}

.explanation-output p {
  color: var(--ink);
  line-height: 1.7;
}

.explanation-output em {
  color: var(--muted);
  font-size: 0.82rem;
}

footer { display: none !important; }

@media (max-width: 880px) {
  .gradio-container { padding: 20px 16px 48px !important; }
  .app-header { align-items: flex-start; flex-direction: column; gap: 20px; }
  .model-note { min-width: 0; }
  .workspace { flex-direction: column !important; }
  .input-panel { position: static; width: 100%; }
  .results-panel { width: 100%; }
}

@media (max-width: 520px) {
  .app-header h1 { font-size: 2.5rem; }
  .input-panel,
  .decision-record,
  .evidence-section,
  .explanation-output > div { padding: 18px !important; }
  .label-line { align-items: flex-start; flex-direction: column; gap: 7px; }
  .signal-list { grid-template-columns: 1fr; }
  .signal-list > div + div { padding-left: 0; border-left: 0; }
  .signal-list > div { padding-right: 0; }
  .section-heading { flex-direction: column; }
}

@media (prefers-reduced-motion: reduce) {
  .primary-action { transition: none !important; }
}

@media (forced-colors: active) {
  .primary-action,
  .context-marker,
  .input-panel,
  .decision-output,
  .evidence-output,
  .explanation-output { border: 1px solid CanvasText !important; }
}
"""


APP_HEAD = """
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="description" content="Review fake-news classifier signals and evidence context." />
"""


if current_process().name == "MainProcess":
    initialize_classifier()
    demo: gr.Blocks | None = build_app()
else:
    demo = None


if __name__ == "__main__":
    if demo is None:
        raise RuntimeError("Gradio app was not initialized in the main process")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        footer_links=[],
        theme="default",
        css=APP_CSS,
        head=APP_HEAD,
    )
