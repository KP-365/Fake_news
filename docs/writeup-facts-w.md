# Write-up Facts: William's Report Sections

This sheet consolidates only values and behaviors backed by committed repository artifacts. Use it for William's report sections on baseline comparison, evidence verification, Google Fact Check coverage, explanation behavior, and the deployed Space. It is not a substitute for the methodology sheet; it is the reporting-safe fact source.

## Artifact map

| Area | Primary committed evidence |
|---|---|
| Baseline and classifier comparison | `docs/results-summary.md`; preserved outputs in `eval_FakeNews.ipynb` |
| MC Dropout calibration and uncertainty | Executed outputs and embedded figures in `eval_MCFakeNews.ipynb`, merged in commit `799f359` |
| Escalation behavior and counts | `evaluation/escalation_results.csv`; `docs/results-summary.md` |
| Google Fact Check coverage and fallback comparison | `google factchek/evaluation/google_escalation_results.csv`; `google factchek/evaluation/google_escalation_summary.txt`; commit `034f8bf` |
| Explanation pipeline | `explain.py`, `pipeline.py`, `app.py`, `eval_faithfulness.ipynb` |
| Explanation faithfulness review | `evaluation/faithfulness_review.csv`; executed `eval_faithfulness.ipynb`; commit `e529b46` |
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

The original DDG-only escalation results are committed in `evaluation/escalation_results.csv`. Recomputing from that CSV confirms the figures summarized in `docs/results-summary.md`:

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

## 2A. Backed Google Fact Check coverage and fallback comparison

Commit `034f8bf` commits the audited fixed-bucket outputs `google factchek/evaluation/google_escalation_results.csv` and `google factchek/evaluation/google_escalation_summary.txt`. The Google-run CSV contains the same 100 row IDs, claims, true labels, classifier labels, and previously committed DDG-only final labels as `evaluation/escalation_results.csv`.

The Google audit establishes:

| Measure | Audited result |
|---|---:|
| Frozen-bucket rows | 100 |
| Google candidate coverage | 0.0% (0/100) |
| Clean zero-candidate no-match responses | 100/100 |
| Google request or API-key errors | 0/100 |
| Usable Google verdicts | 0.0% (0/100) |
| Google agreement with true labels | Not applicable (0/0) |

Every row has `google_candidate_count = 0` and `google_reason = no Google fact-check match`, while every `google_error` field is empty. The zero coverage is therefore an audited clean no-match result, not an HTTP, request, or API-key failure. With no usable Google verdicts, agreement cannot be calculated and must not be reported as 0%.

The three-way accuracy comparison on the frozen bucket is:

| Decision path | Correct | Accuracy |
|---|---:|---:|
| Classifier-only | 76/100 | 76.0% |
| Committed DDG-only final | 64/100 | 64.0% |
| Google-first hybrid final | 64/100 | 64.0% |

Because Google returned no candidates, all 100 rows fell through to a new live DDG + DeBERTa pass. Google therefore made no measured contribution to the hybrid result. The two committed DDG passes have identical aggregate accuracy but different verdict distributions:

| Verdict | Original DDG-only pass (`escalation_results.csv`) | Google-run DDG fallback (`google_escalation_results.csv`) |
|---|---:|---:|
| Supported | 42 | 30 |
| Refuted | 32 | 12 |
| Insufficient | 26 | 58 |
| **Final correct** | **64/100** | **64/100** |

The shift from **42/32/26** to **30/12/58** supported/refuted/insufficient verdicts is committed evidence of live-retrieval verdict instability, even though aggregate final accuracy remained 64.0% in both passes. It supports treating live retrieval and NLI as contextual evidence rather than a reproducible automatic override.

The Google-run CSV has **58** rows with `source=none`, all with an `insufficient` verdict and `reason = DDG evidence evaluated with DeBERTa NLI`. In this artifact, `source=none` does not mean that no DDG evidence was retrieved: `google factchek/verify.py` assigns that source when retrieved evidence has been evaluated but neither entailment nor contradiction clears the NLI verdict threshold. A genuine no-evidence case would instead record `reason = no DDG evidence retrieved`.

These findings are limited to the frozen 100-row low-confidence WELFake bucket and the query-time API responses recorded in commit `034f8bf`; they do not establish Google Fact Check coverage for other claims, query formulations, datasets, or times.

## 3. Explanation pipeline behavior and faithfulness review

These behaviors and results are backed by `explain.py`, `pipeline.py`, `app.py`, the executed `eval_faithfulness.ipynb`, and `evaluation/faithfulness_review.csv`:

- `pipeline.py` chains classifier inference, **30 MC Dropout passes**, DDG/NLI verification, and the optional explanation call.
- `explain.py` sends only seven structured signals: classifier label, classifier confidence, MC Dropout uncertainty, NLI verdict, maximum entailment, maximum contradiction, and evidence count.
- The explanation model is `claude-haiku-4-5`, with `max_tokens=220` and `temperature=0`.
- The prompt fixes the classifier label and instructs the model not to reclassify it or imply a different label is more accurate.
- If the NLI verdict conflicts with the label, the prompt requires the explanation to describe the conflict as context and state that it does not change the classifier label.
- `app.py` keeps the classifier label final and marks NLI evidence as context only.
- `app.py` accepts an optional per-request Anthropic key, clears the key output on every return path, and returns an explanation-unavailable note without a key while preserving classifier and verification results.
- The executed `eval_faithfulness.ipynb` applies the manual 10-row review workflow and exports the completed `evaluation/faithfulness_review.csv` committed in `e529b46`.

### Completed 10-row faithfulness result

| Review outcome | Rows | Rate |
|---|---:|---:|
| Matches supplied structured signals (`yes`) | 8 | 80% |
| Does not match supplied structured signals (`no`) | 2 | 20% |
| **Overall reviewed** | **10** | **100%** |

The two rows marked `no` document distinct failures:

| Source row | Documented failure |
|---:|---|
| 8 | The classifier label was fake while the NLI verdict was `supported` with entailment 0.98. The explanation claimed that the evidence aligned with fake, inverting the verdict's direction. |
| 94 | The explanation attributed the classifier decision to retrieved evidence items that the classifier never saw. |

The reporting-safe result is therefore **8/10 faithful (80%)** under the review's matches-signals criterion. This is a small sample judged by a single reviewer, so it does not establish population-level faithfulness or inter-rater reliability. Scale-up should use more explanations, multiple independent reviewers, a prespecified rubric, and an agreement measure.

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

The MC summary results and arrays, the Google fixed-bucket audit, and the completed 10-row explanation-faithfulness review are now backed by committed artifacts. The remaining reporting boundaries are:

1. **Baseline comparison CSV.** Baseline results are preserved in `docs/results-summary.md` and the `eval_FakeNews.ipynb` output, but no dedicated baseline CSV is committed.
2. **Permanent output transcript for the deployed Space run.** The verified endpoint result is documented in `docs/results-summary.md`; no separate committed log file stores the full response payload. Report the summary values only.
3. **Stable live DDG evidence URLs for future runs.** `docs/results-summary.md` warns that DDG search results are live web data and can change between runs.
4. **Current live Space RUNNING state in repository evidence.** The Space can be observed externally, but the repository does not commit a runtime-status artifact. Use the URL and documented endpoint verification, not a claimed repository-backed runtime state.
