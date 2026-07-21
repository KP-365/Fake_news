# Results Summary

This page consolidates the quantitative results currently preserved in the repository. It is
intended as a factual source for the report write-up. Values come from the committed outputs of
`eval_FakeNews.ipynb` and the 100 rows in `evaluation/escalation_results.csv`.

## 1. Held-out WELFake classifier evaluation

The evaluation recreates the training notebook's deterministic, stratified 70/15/15 split with
seed 42. The held-out test set contains 9,398 articles: 5,193 real and 4,205 fake. None of these
articles were used for training.

### RoBERTa-LoRA metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9957 |
| Macro precision | 0.9958 |
| Macro recall | 0.9956 |
| Macro F1 | 0.9957 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Real | 0.9952 | 0.9971 | 0.9962 | 5,193 |
| Fake | 0.9964 | 0.9941 | 0.9952 | 4,205 |

The confusion matrix uses true labels as rows and predicted labels as columns:

```text
[[5178,   15],
 [  25, 4180]]
```

The classifier made 40 errors: 15 real articles were labelled fake and 25 fake articles were
labelled real.

### Baseline comparison

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| RoBERTa-LoRA | 0.9957 | 0.9957 |
| TF-IDF + logistic regression | 0.9526 | 0.9521 |
| Majority-class baseline | 0.5526 | 0.3559 |

RoBERTa-LoRA exceeds the TF-IDF baseline by 4.31 percentage points in accuracy and 4.36 points
in macro F1. The majority-class baseline always predicts the most common training label and is
included as a lower-bound comparison.

## 2. MC Dropout calibration and uncertainty

`eval_MCFakeNews.ipynb` contains the implemented 30-pass MC Dropout evaluation, predictive
entropy and mutual-information diagnostics, a rejection curve, a 15-bin reliability diagram,
and ECE calculations for MC-averaged and deterministic confidence.

The committed notebook does **not** contain executed outputs for the MC inference, calibration,
or ECE cells. Consequently, there is no numerical MC Dropout ECE, deterministic ECE, entropy,
or accuracy-versus-coverage result available to quote. Reporting a calibration number before
that notebook is fully executed and its outputs are committed would be inventing a result.

## 3. NLI escalation on the 100 lowest-confidence articles

`evaluation/escalation_results.csv` contains the 100 WELFake test articles selected as least
confident by MC Dropout. The classifier was correct on 76 articles before escalation. DDG
retrieval plus DeBERTa NLI produced 42 `supported`, 32 `refuted`, and 26 `insufficient`
verdicts.

A verdict changes the label only when `supported` maps it to real or `refuted` maps it to fake.
`insufficient` keeps the classifier label.

| Verdict | Rows | Classifier accuracy | Actual overrides | Wrong predictions fixed | Correct predictions broken | Final accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Supported | 42 | 73.8% (31/42) | 14 | 7 | 7 | 73.8% (31/42) |
| Refuted | 32 | 71.9% (23/32) | 20 | 4 | 16 | 34.4% (11/32) |
| Insufficient | 26 | 84.6% (22/26) | 0 | 0 | 0 | 84.6% (22/26) |
| **Overall** | **100** | **76.0% (76/100)** | **34** | **11** | **23** | **64.0% (64/100)** |

### Honest interpretation

This experiment is a negative result for automatic NLI override. Escalation reduced accuracy on
the low-confidence bucket from 76% to 64%, a loss of 12 percentage points. Across 34 actual
label changes, NLI fixed 11 classifier mistakes but broke 23 predictions that were originally
correct.

The `supported` verdicts were neutral overall: seven fixes and seven breaks. The `refuted`
verdicts were the main failure mode: four fixes versus 16 breaks, reducing that group's accuracy
from 71.9% to 34.4%. The `insufficient` group had the best classifier-only accuracy at 84.6%, and
its labels were correctly left unchanged by the escalation rule.

These results do not show that evidence retrieval or NLI is useless. They show that the current
unthresholded verdict-to-label mapping is unsafe as an automatic override. Web snippets can be
irrelevant, ambiguous, or about a related claim, and contradiction scores do not by themselves
establish article-level falsity. The current output is better suited to human review or to a
future escalation policy with evidence-quality checks and validated score thresholds.

## 4. Reporting boundaries

- All classifier metrics are in-dataset results on the held-out WELFake split, not LIAR or a
  cross-dataset test.
- The escalation experiment covers the 100 least-confident test articles, not the full 9,398.
- DDG search results are live web data and can change between runs.
- No numerical calibration result is currently preserved in the MC Dropout notebook outputs.
- The live ZeroGPU Space is available at https://wf1212-fake-news-detector.hf.space. A verified
  `gradio_client` call to the named `/analyze` endpoint on 20 July 2026 completed in 64.4 seconds
  wall-clock and returned REAL at 99.57% confidence, Very stable prediction stability, and an NLI
  verdict of Supported. Runtime includes live DDG retrieval and two ZeroGPU allocations.
- The completed 10-row, single-reviewer explanation-faithfulness review marked 8/10 explanations
  faithful (80%). The two failures were the row-8 NLI verdict inversion and row 94's attribution of
  the classifier decision to retrieved evidence that the classifier never saw
  (`evaluation/faithfulness_review.csv`, commit `e529b46`). Treat this as a small-sample descriptive
  result rather than a population estimate or multi-reviewer agreement measure.
