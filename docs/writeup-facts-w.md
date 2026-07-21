# Write-up Facts: William's Report Sections

This sheet consolidates only values and behaviors backed by committed repository artifacts. Use it for William's report sections on baseline comparison, explanation behavior, and the deployed Space. It is not a substitute for the methodology sheet; it is the reporting-safe fact source.

## Artifact map

| Area | Primary committed evidence |
|---|---|
| Baseline and classifier comparison | `docs/results-summary.md`; preserved outputs in `eval_FakeNews.ipynb` |
| MC Dropout calibration and uncertainty | Executed outputs and embedded figures in `eval_MCFakeNews.ipynb`, merged in commit `799f359` |
| Escalation behavior and counts | `evaluation/escalation_results.csv`; `docs/results-summary.md` |
| Explanation pipeline | `explain.py`, `pipeline.py`, `app.py`, `eval_faithfulness.ipynb` |
| Deployed Space | `space/README.md`, `app.py`, `requirements.txt`, `docs/results-summary.md` |
| Shared training/inference definitions | `docs/methodology-facts.md`, `predict.py` |

## 1. Baseline comparison results

All figures below are for the same deterministic, stratified, seeded WELFake held-out test split. `docs/methodology-facts.md` and `docs/results-summary.md` both state a 70/15/15 split with seed 42, a held-out test size of 9,398 articles, and test support of 5,193 real and 4,205 fake.

| Model | Accuracy | Macro F1 | Backing |
|---|---:|---:|---|
| RoBERTa-LoRA | 0.9957 | 0.9957 | `docs/results-summary.md`; preserved `eval_FakeNews.ipynb` output |
| TF-IDF + logistic regression | 0.9526 | 0.9521 | `docs/results-summary.md`; preserved `eval_FakeNews.ipynb` output |
| Majority-class baseline | 0.5526 | 0.3559 | `docs/results-summary.md`; preserved `eval_FakeNews.ipynb` output |

RoBERTa-LoRA is 4.31 percentage points above the TF-IDF baseline in accuracy and 4.36 percentage points above it in macro F1, as recorded in `docs/results-summary.md`.

The RoBERTa-LoRA confusion matrix is:

```text
[[5178,   15],
 [  25, 4180]]
```

Rows are true labels in the order real, fake. This records 40 errors: 15 real articles labelled fake and 25 fake articles labelled real.

### Baseline implementation details from `eval_FakeNews.ipynb`

- TF-IDF uses `max_features=30000`, `ngram_range=(1, 2)`, and `min_df=2`.
- Logistic regression uses `max_iter=1000`, `random_state=42`, and `solver="liblinear"`.
- The majority baseline is `DummyClassifier(strategy="most_frequent")`.

## 1A. MC Dropout calibration and uncertainty

The executed outputs merged in commit `799f359` back the following results:

| Inference mode | Accuracy | Macro F1 |
|---|---:|---:|
| MC Dropout, 30 passes | 0.9967 | 0.9967 |
| Deterministic | 0.9957 | 0.9957 |

The MC Dropout confusion matrix is:

```text
[[5175,   18],
 [  13, 4192]]
```

The executed uncertainty summaries are:

| Measure | Overall mean | Correct predictions | Incorrect predictions |
|---|---:|---:|---:|
| Predictive entropy | 0.0347 | 0.0333 | 0.4816 |
| Mutual information | 0.0015 | 0.0013 | 0.0468 |

The 15-bin expected calibration errors are **0.0055 for MC Dropout** and **0.0028 for deterministic confidence**. Since lower ECE is better, MC Dropout did not improve calibration in this run, despite its higher accuracy and macro F1.

The notebook embeds three relevant visual artifacts: the MC confusion-matrix figure; a combined correct-vs-incorrect predictive-entropy boxplot and rejection curve; and the 15-bin reliability diagram comparing MC with deterministic confidence. All values and figures are preserved in executed cells of `eval_MCFakeNews.ipynb`.

The four source arrays are now committed under `evaluation/mc_arrays/`: `mean_probabilities.npy` with shape `(9398, 2)`, plus `predictive_entropy.npy`, `expected_entropy.npy`, and `true_labels.npy` with shape `(9398,)`. Loading these arrays independently reproduces predictive-entropy mean 0.0347, mutual-information mean 0.0015, argmax accuracy 0.9967, and confusion matrix `[[5175, 18], [13, 4192]]`.

## 2. Evidence-verification and escalation numbers

The only committed escalation CSV is `evaluation/escalation_results.csv`. Recomputing from that CSV confirms the figures summarized in `docs/results-summary.md`:

| Measure | Exact count | Reported accuracy |
|---|---:|---:|
| Rows reviewed | 100 | Not applicable |
| Classifier-only correct | 76 | 76.0% |
| Final label correct | 64 | 64.0% |
| Actual label changes | 34 | Not applicable |
| Classifier mistakes fixed | 11 | Not applicable |
| Correct predictions broken | 23 | Not applicable |

The CSV itself contains 66 true-real and 34 true-fake rows, 62 classifier-real and 38 classifier-fake rows, and 56 final-real and 44 final-fake rows. These counts are not repeated in `docs/results-summary.md` but are directly derivable from the committed CSV.

| Verdict | Rows | Classifier correct before override | Final correct after override | Actual changes | Fixes | Breaks |
|---|---:|---:|---:|---:|---:|---:|
| Supported | 42 | 31 | 31 | 14 | 7 | 7 |
| Refuted | 32 | 23 | 11 | 20 | 4 | 16 |
| Insufficient | 26 | 22 | 22 | 0 | 0 | 0 |
| **Overall** | **100** | **76** | **64** | **34** | **11** | **23** |

This is a negative result for automatic NLI override: accuracy falls by 12 percentage points, from 76.0% to 64.0%. `refuted` is the main failure mode, moving that group from 23/32 correct to 11/32 correct.

## 3. Explanation pipeline behavior

These behaviors are backed by `explain.py`, `pipeline.py`, `app.py`, and `eval_faithfulness.ipynb`:

- `pipeline.py` chains classifier inference, **30 MC Dropout passes**, DDG/NLI verification, and the optional explanation call.
- `explain.py` sends only seven structured signals: classifier label, classifier confidence, MC Dropout uncertainty, NLI verdict, maximum entailment, maximum contradiction, and evidence count.
- The explanation model is `claude-haiku-4-5`, with `max_tokens=220` and `temperature=0`.
- The prompt fixes the classifier label and instructs the model not to reclassify it or imply a different label is more accurate.
- If the NLI verdict conflicts with the label, the prompt requires the explanation to describe the conflict as context and state that it does not change the classifier label.
- `app.py` keeps the classifier label final and marks NLI evidence as context only.
- `app.py` accepts an optional per-request Anthropic key, clears the key output on every return path, and returns an explanation-unavailable note without a key while preserving classifier and verification results.
- `eval_faithfulness.ipynb` creates a manual 10-row review workflow using `np.random.default_rng(42)` and exports `evaluation/faithfulness_review.csv`.

## 4. Deployed Space facts

- Space URL: `https://wf1212-fake-news-detector.hf.space`.
- Space metadata in `space/README.md` sets `sdk: gradio`, `sdk_version: 6.20.0`, `python_version: 3.12.12`, and `app_file: app.py`.
- The Space README describes a ZeroGPU demo that classifies with RoBERTa-LoRA, reports MC Dropout stability, and shows DDG plus DeBERTa evidence as context that never overrides the classifier label.
- The public API event is named `analyze` in `app.py`, with `api_visibility="public"`.
- `app.py` bounds submitted articles to the first 5,000 characters.
- The Space README says the optional Anthropic key is password-masked, never added to the environment, and cleared after analysis.
- `docs/results-summary.md` records a verified `gradio_client` call to the named `/analyze` endpoint on 20 July 2026:
  - 64.4 seconds wall-clock;
  - final classifier label REAL;
  - confidence 99.57%;
  - prediction stability Very stable;
  - NLI verdict Supported;
  - runtime includes live DDG retrieval and two ZeroGPU allocations.

## 5. Claims that currently lack committed artifact backing

The MC summary results, embedded figures, and four raw `.npy` arrays are now committed artifacts. The remaining reporting boundaries are:

1. **Explanation faithfulness rate or human-evaluation result.** `eval_faithfulness.ipynb` provides the workflow, but no completed `evaluation/faithfulness_review.csv` is committed. `docs/results-summary.md` explicitly states that no explanation-faithfulness or human-evaluation result has been completed.
2. **Baseline comparison CSV.** Baseline results are preserved in `docs/results-summary.md` and the `eval_FakeNews.ipynb` output, but no dedicated baseline CSV is committed under `evaluation/`. The only committed evaluation CSV is `escalation_results.csv`.
3. **Permanent output transcript for the deployed Space run.** The verified endpoint result is documented in `docs/results-summary.md`; no separate committed log file stores the full response payload. Report the summary values only.
4. **Stable live DDG evidence URLs for future runs.** `docs/results-summary.md` warns that DDG search results are live web data and can change between runs.
5. **Current live Space RUNNING state in repository evidence.** The Space can be observed externally, but the repository does not commit a runtime-status artifact. Use the URL and documented endpoint verification, not a claimed repository-backed runtime state.
