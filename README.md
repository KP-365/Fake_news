# A Three-Layer Framework for Fake News Detection, Verification, and Explanation

ECS7036P Group 10 project. Combines machine learning, external fact verification, and
explainable AI into a single pipeline that classifies a claim, reports a calibrated
confidence, surfaces conflicting fact-checks, and returns a human-readable explanation.

**Deadline: 21 July 2026** (tentative — confirm against QMplus/module page).

## Team

| Name | Email | Focus |
|---|---|---|
| Kayleb Elorm Aston Parkes | ec251170@qmul.ac.uk | Layer 1 — Classification |
| Eric Kamalendran | ec251192@qmul.ac.uk | Layer 2 — Verification |
| William Finlay McKie | ec251168@qmul.ac.uk | Layer 3 — Explanation & UI |

See [TASKS.md](TASKS.md) for the full task tracker and [GitHub Issues](../../issues) /
[Project board](../../projects) for live status.

## Architecture

1. **Layer 1 — Classification** (`src/classifier/`): BERT-base-uncased fine-tuned with
   LoRA (Hugging Face PEFT) classifies a claim as real or fake. Monte Carlo Dropout
   estimates prediction uncertainty so the model can flag low-confidence claims instead
   of forcing a binary decision.
2. **Layer 2 — Verification** (`src/verification/`): Queries the Google Fact Check
   Tools API (PolitiFact, Snopes, etc.) for existing fact-checks and flags disagreements
   between the classifier and retrieved evidence.
3. **Layer 3 — Explanation** (`src/explanation/`): Generates a natural-language
   explanation combining the linguistic signal, verification evidence, and uncertainty
   estimate, surfaced through a Gradio UI (`src/app/`) deployed on Hugging Face Spaces.

## Resources

- **Dataset (primary):** [LIAR](https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset/data) — 12,836 labelled political statements from PolitiFact
- **Dataset (backup):** [WELFake](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) — 72,134 labelled news articles
- **Pretrained model:** BERT-base-uncased via Hugging Face Transformers
- **Fine-tuning:** LoRA via Hugging Face PEFT
- **Fact checking:** Google Fact Check Tools API
- **Compute:** Google Colab (T4 GPU, free tier)
- **UI/hosting:** Gradio + Hugging Face Spaces
- **Language:** Python 3.10+

## Evaluation

- **Layer 1:** per-class precision/recall, macro-F1, confusion matrix on LIAR test set
  (WELFake as second source); reliability diagram and Expected Calibration Error (ECE)
  vs. an uncalibrated softmax baseline; accuracy-on-retained under selective deferral.
- **Layer 2:** coverage of test claims and rate of classifier/fact-check disagreement.
- **Layer 3:** sample-based faithfulness of explanations to the underlying signals.
- **Overall:** success = the deployed Gradio demo classifies a claim, reports calibrated
  confidence, surfaces conflicting fact-checks, and returns a readable explanation.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Repository layout

```
src/
  classifier/    Layer 1 — BERT-LoRA classifier + Monte Carlo Dropout
  verification/  Layer 2 — Google Fact Check API + conflict detection
  explanation/   Layer 3 — explanation agent
  baseline/      Baseline models for comparison
  app/           Gradio UI + Hugging Face Spaces deployment
evaluation/      Metrics: F1, confusion matrix, reliability diagram, ECE
data/            raw/ and processed/ datasets (gitignored, not committed)
notebooks/       Exploratory / Colab training notebooks
docs/            Proposal, report drafts, slides
```

## References

See the [project proposal](docs/) for the full related-work and reference list
(Devlin et al. 2019 — BERT; Hu et al. — LoRA; Wang 2017 — LIAR; Thorne et al. 2018 —
FEVER; Gunning & Aha 2019 — XAI; Gal & Ghahramani 2016 — MC Dropout; Guo et al. 2017 —
calibration; Geifman & El-Yaniv 2017 — selective classification).
