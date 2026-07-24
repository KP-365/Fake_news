# Fake-News Classification, Verification, and Explanation

**Live demo:** [Click Here](https://wf1212-fake-news-detector.hf.space) (ZeroGPU; the Claude explanation is
optional and uses a bring-your-own Anthropic key).

**NOTE - THE APPLICATION IS ONLY VISIBLE IN LIGHT MODE. PLEASE CHANGE YOUR DEVICE TO LIGHT MODE TO HAVE THE BEST EXPERIENCE.**

## Code submission quick start

- **Dataset:** [WELFake on Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- **Setup:** `pip install -r requirements.txt`
- **Classifier:** `python predict.py "Article text or headline"`
- **End-to-end pipeline:** `python pipeline.py "Article text or headline"`
- **Evaluation statistics:** `python evaluation/report_statistics.py`
- **Trained model artifacts:** `models/roberta-trained-welfake/`

This repository implements a **three-stage research pipeline**:

1. **classify** a news article as real or fake with a fine-tuned RoBERTa model;
2. **verify** every input in parallel with web-evidence retrieval and natural-language inference
   (NLI), presented as supporting context only;
3. **explain** the fixed classifier decision from structured model signals only.

The trained checkpoint, runnable scripts, local Gradio interface, and public Hugging Face Space
are all available from this repository.

---

## Implemented pipeline

### 1. RoBERTa-LoRA classifier
`scaffold_FakeNews_finn's_training.ipynb` fine-tunes `roberta-base` with LoRA adapters on WELFake.
It cleans and combines each article's title and body, then creates a deterministic, stratified
70/15/15 train/validation/test split with **seed 42**. The class mapping is **0 = real, 1 = fake**.

Training uses a maximum sequence length of **256**, batch size **32**, validation-loss early
stopping, and at most **three epochs**. The best adapter is restored before final evaluation and
saved with the separately trained classifier head under `models/roberta-trained-welfake/`.
`predict.py` loads that local checkpoint and returns one label plus its softmax confidence.

### 2. MC Dropout uncertainty and offline escalation
`eval_MCFakeNews.ipynb` performs **30 stochastic forward passes** with dropout active. It uses the
maximum class probability from the mean prediction as confidence, and also computes entropy,
mutual information, calibration, and accuracy-versus-coverage diagnostics.

`evaluation/eval_escalation.py` applies the same MC Dropout procedure to the 9,398-article WELFake
test split and selects the **100 lowest-confidence** articles for an offline escalation experiment.
This is **not** a production confidence gate: the deployed pipeline runs retrieval and NLI
verification for every input, presents the verdict as supporting context, and keeps the classifier
label final. For the offline experiment, each selected article's title is passed to `verify.py`; if
the title is empty, the first 300 characters of the combined content are used instead.

### 3. DDG retrieval and DeBERTa NLI
`verify.py` searches the web through `ddgs`, using a 10-second DDG timeout inside a 15-second
hard-capped worker process. Retrieved snippets are compared with the claim by
`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. The offline escalation experiment uses this
counterfactual mapping:

- `refuted → fake`
- `supported → real`
- `insufficient → keep the RoBERTa classifier label`

The escalation evaluator writes article-level outputs to `evaluation/escalation_results.csv` and
prints classifier-only accuracy, escalated accuracy, and the NLI verdict distribution for the
low-confidence bucket.

### 4. Numbers-only Anthropic explanation
`explain.py` sends **only** these structured values to Anthropic: classifier label and confidence;
MC Dropout uncertainty; NLI verdict; maximum entailment and contradiction scores; evidence count.
It does **not** send the article, claim, evidence text, or URLs. Its prompt treats the supplied
label as final and explicitly tells the model to explain it in plain English without reclassifying
or second-guessing it. `ANTHROPIC_API_KEY` must be set in the environment. The explanation model is
`claude-haiku-4-5`.

---

## Proposal deviations

The original proposal describes a different target. The material deviations are explicit:

| Proposal | Implemented repository |
|---|---|
| LIAR as the primary training/evaluation dataset | WELFake is the sole training and reported test dataset |
| `bert-base-uncased` classifier | `roberta-base` with LoRA |
| Google Fact Check Tools API and named fact-check sites | General DDG web retrieval followed by DeBERTa NLI |
| Claude Sonnet explanation layer | Anthropic `claude-haiku-4-5`, using numbers only |
| Gradio UI and Hugging Face Spaces deployment | Public ZeroGPU Gradio Space deployed |

Results in this repository therefore measure **in-dataset WELFake performance**. They should not be
presented as LIAR results or as evidence of cross-dataset generalisation.

---

## Validated classifier result

`eval_FakeNews.ipynb` recreates the unseen WELFake split and reports:

- RoBERTa-LoRA accuracy: **0.9957**
- RoBERTa-LoRA macro-F1: **0.9957**
- TF-IDF + logistic-regression macro-F1: **0.9521**
- majority-class baseline macro-F1: **0.3559**
- confusion matrix: `[[5178, 15], [25, 4180]]`

The completed single-reviewer faithfulness study is committed in
`evaluation/faithfulness_review.csv`: **8 of 10** explanations matched their structured inputs.

---

## Notebook section-by-section maps

Each notebook below is mapped section-by-section — what the code does, **why** it's there, and
**what it gave us** — so the experimental story is easy to follow (and to defend in the viva).

### [scaffold_FakeNews_finn's_training.ipynb](https://github.com/KP-365/Fake_news/blob/main/scaffold_FakeNews_finn's_training.ipynb) — training (master scaffold)

The master notebook that produces the trained checkpoint. Training lives in the early sections;
the later layers are scaffolded and were extracted into standalone scripts (`verify.py`,
`explain.py`, `app.py`).

| Section | What it does |
|---|---|
| **0. Setup** | Imports `transformers`, `peft`, `torch` |
| **1. Data** | Loads WELFake via `kagglehub`, cleans title+body, inspects it (e.g. the `(Reuters)` source-leak check), builds the stratified 70/15/15 split (seed 42) |
| **2. Layer 1 — Classification (RoBERTa-LoRA)** | Wraps `roberta-base` with LoRA adapters (**109.8M → 294,912 trainable params**) and trains with early stopping |
| **3. Layer 1 — MC Dropout** | Saves the classifier head; sets up the uncertainty machinery |
| **4. Layer 1 — Evaluation** | F1, confusion matrix, calibration |
| **5–10** | Layer 2 (fact-check / verification), Layer 3 (explanation), Gradio + HF Spaces, end-to-end & faithfulness — scaffolded here, implemented in the standalone scripts |

**Why / what it gave us:** this notebook *produces* `models/roberta-trained-welfake/` — the artifact
every other notebook and script loads. LoRA cuts trainable parameters from ~109.8M to ~294,912
(a tiny adapter), which is what makes the checkpoint small and training cheap. *(Note: an early data
cell still references `train_liar_df`, a leftover from the original LIAR proposal — the implemented
pipeline uses WELFake, per the proposal-deviations table above.)*

### [Trained_vs_base.ipynb](https://github.com/KP-365/Fake_news/blob/main/Trained_vs_base.ipynb) — ablation: fine-tuned vs base model

Does **not** train anything. Loads the trained checkpoint, spins up a fresh un-fine-tuned
`roberta-base` (same architecture and tokenizer, random classification head), and evaluates both on
the same test set to isolate the effect of fine-tuning. All cells are code with inline comments.

| Cell | What it does |
|---|---|
| **1. Setup** | `git clone` + `pip install` + torchvision fix |
| **2. Load trained model** | `load_model()` → the fine-tuned model |
| **3. Test split** | WELFake cleaning + stratified 70/15/15 (seed 42) |
| **4. Load base model** | fresh `roberta-base` with a random head — the "before fine-tuning" reference |
| **5. Evaluate trained** | `evaluate_model` / `score` helpers → classification report |
| **6. Evaluate base** | same pipeline on the base model → classification report |
| **7. Compare** | metrics table (accuracy, precision, recall, F1, accuracy gain) + paired confusion matrices |

**Why / what it gave us:** a controlled ablation where fine-tuning is the *only* difference between
the two models. It proves the LoRA fine-tuning is what creates the capability — the base model
(random head) scores near chance, the fine-tuned model does not. *This is an ablation, not a
competitive baseline* (those live in `eval_FakeNews.ipynb §5`).

### [eval_FakeNews.ipynb](https://github.com/KP-365/Fake_news/blob/main/eval_FakeNews.ipynb) — baseline evaluation

Single deterministic forward pass — establishes raw accuracy and compares against simple baselines.

| Section | What it does |
|---|---|
| **1–3** | Setup, load checkpoint, rebuild the held-out split |
| **4. Evaluate** | Batched inference → accuracy, macro-F1, per-class metrics, confusion matrix |
| **5. Baselines** | TF-IDF + Logistic Regression and majority-class; metrics table + macro-F1 bar chart |
| **6. Sample predictions** | 10 seeded articles, true vs predicted + confidence |

**Why / what it gave us:** the headline accuracy/macro-F1 (**0.9957**) and proof the fine-tuned
transformer beats simple word-frequency baselines (TF-IDF+LR macro-F1 0.9521; majority 0.3559) — the
"does it work, and is it worth it?" evidence.

### [eval_MCFakeNews.ipynb](https://github.com/KP-365/Fake_news/blob/main/eval_MCFakeNews.ipynb) — MC Dropout / uncertainty evaluation

Builds on the baseline to quantify *uncertainty* and prove it's useful. Saves every output to
`MC results/` (see [`MC results/README.md`](MC%20results/README.md)).

| Section | What it does | Saves to `MC results/` |
|---|---|---|
| **1. Setup** | Clone + install + torchvision fix | — |
| *(auth)* | Hugging Face login (avoids download rate limits) | — |
| **2. Load checkpoint** | `load_model()` | — |
| **3. Test split** | WELFake cleaning + 70/15/15 (seed 42) | — |
| *(setup)* | Create the `MC results/` folder | *(folder)* |
| **4. MC-Dropout inference** | 30 passes → entropy + mutual information | — |
| **5. Evaluate** | Accuracy, macro-F1, confusion matrix | `confusion_matrix.png`, `classifier_metrics.json` |
| **6. Uncertainty vs errors** | Entropy boxplot + rejection curve | `uncertainty_rejection.png`, `uncertainty_summary.json`, `rejection_curve.npz` |
| **7. Calibration** | Reliability diagram + ECE | `reliability_diagram.png`, `calibration.json` |
| **8. Sample predictions** | 10 seeded articles with uncertainty | `sample_predictions.csv` |
| **9. Save & push** | Per-article CSV + raw arrays; `git push` | `per_article_results.csv`, `mc_arrays.npz` |

**Why / what it gave us:** per-article confidence and uncertainty (the core contribution), evidence
that entropy separates correct from incorrect predictions (the rejection curve), and proof the
confidences are well-calibrated (ECE) — the basis for the escalation experiment.

---

## Setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The first classifier or verifier run downloads its Hugging Face base model. The LoRA adapter,
classifier head, and tokenizer are already stored under `models/`.

## Run the scripts

**Classify one article or headline**
```bash
python3 predict.py "Federal Reserve holds interest rates steady amid mixed economic data"
```
Output is the fixed label and confidence, e.g. `real (confidence: 99.57%)`.

**Retrieve evidence and run NLI**
```bash
python3 verify.py "Federal Reserve holds interest rates steady amid mixed economic data"
```
Requires internet access; prints JSON with the verdict, NLI scores, and retrieved evidence.

**Run the hardcoded explanation example**
```bash
cp .env.example .env
# Edit .env and fill in ANTHROPIC_API_KEY, then run:
python3 explain.py
```
`explain.py` loads `.env` for its CLI example (`.env` is excluded from Git). For application code,
import `explain_decision(...)` and provide the seven structured values listed above. The CLI runs
one hardcoded example; it does not accept article text.

**Run the end-to-end command**
```bash
python3 pipeline.py "Federal Reserve holds interest rates steady amid mixed economic data"
```
Runs classification, 30-pass MC Dropout, DDG + NLI verification, and the numbers-only explanation.
The classifier label remains final; NLI is explanation context only. Without an Anthropic key, the
command still prints every non-explanation result and a clear setup note.

**Launch the local Gradio interface**
```bash
python3 app.py
```
Open http://127.0.0.1:7860. Loads the classifier once at startup; presents the fixed label,
plain-language stability with the raw uncertainty value, context-only evidence links, NLI verdict,
and Claude explanation. An optional accordion accepts a per-request Anthropic key and clears it
after analysis.

## Run the evaluation notebooks

Use a Colab GPU runtime, then **Runtime → Run all**.

- `eval_FakeNews.ipynb` — deterministic held-out classifier evaluation, per-class metrics, confusion
  matrix, ten sample predictions, and TF-IDF/majority-class baselines.
- `eval_MCFakeNews.ipynb` — full MC Dropout uncertainty, calibration, and selective-retention
  diagnostics. Saves all outputs to `MC results/`.
- `eval_escalation.ipynb` — clones/updates the repo, installs Colab dependencies, prints the exact
  Git commit and CUDA status, and runs `evaluation/eval_escalation.py`. Intentionally long: 30 passes
  over the full test split plus 100 rate-limited web/NLI checks.

For a local escalation run after setup:
```bash
python3 evaluation/eval_escalation.py
```

---

## Repository layout

```
predict.py                                   RoBERTa-LoRA command-line inference
verify.py                                    DDG retrieval and DeBERTa NLI verification
explain.py                                   numbers-only Anthropic explanation
pipeline.py                                  end-to-end command with fixed classifier label
app.py                                       local Gradio interface
scaffold_FakeNews_finn's_training.ipynb      WELFake training notebook (master scaffold)
Trained_vs_base.ipynb                        ablation: fine-tuned vs raw base transformer
eval_FakeNews.ipynb                          held-out classifier and baseline evaluation
eval_MCFakeNews.ipynb                        MC Dropout / calibration evaluation
eval_escalation.ipynb                        Colab runner for escalation evaluation
evaluation/eval_escalation.py                low-confidence bucket and NLI escalation
evaluation/escalation_results.csv            committed article-level escalation results
evaluation/faithfulness_review.csv           committed single-reviewer faithfulness study
models/roberta-trained-welfake/              tracked adapter, classifier head, tokenizer
MC results/                                   MC Dropout evaluation outputs (see its README)
docs/                                        citations, research notes, results summaries, drafts
requirements.txt                             Python dependencies
TASKS.md                                     implementation-status checklist
```

See `TASKS.md` for completed work and remaining gaps, and
[`MC results/README.md`](MC%20results/README.md) for the Monte Carlo evaluation outputs.
