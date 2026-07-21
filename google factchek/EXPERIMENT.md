# Google Fact Check Experiment Paths

This folder contains two Google-first verification runners with different purposes. They share `verify.py`, but they do not evaluate the same workflow.

## Design

Google Fact Check is queried before DDG/DeBERTa. A sufficiently similar, unambiguous Google rating produces `supported` or `refuted`; otherwise verification falls back to live DDG retrieval and DeBERTa NLI. The final-label rule is:

- `supported` -> `real`;
- `refuted` -> `fake`;
- `insufficient` -> the existing classifier label.

The API key is read from the process environment and is never written to result files.

## Runner 1: Eric's full pipeline

`evaluation/eval_escalation.py` reconstructs the WELFake split, loads the classifier, performs 30-pass MC Dropout across the held-out set, selects the 100 lowest-confidence articles, and runs Google-first verification. This is the full model-to-escalation experiment and requires the model, Torch, Kaggle data, and `GOOGLE_FACTCHECK_API_KEY`.

Run from this folder:

```bash
export GOOGLE_FACTCHECK_API_KEY="YOUR_KEY"
python evaluation/eval_escalation.py
```

## Runner 2: fixed-bucket Google evaluation

`evaluation/eval_google_bucket.py` does not rerun classification, MC Dropout, split construction, or Kaggle download. It reads the parent repository's committed `evaluation/escalation_results.csv` and uses its 100 rows as a frozen bucket. Each row contributes:

- `claim` when present, otherwise `text_snippet`;
- `classifier_label`;
- `true_label`;
- the committed DDG-only `final_label` for comparison.

The runner hides CUDA for its process. Google requests do not load Torch or a model. Only rows that fall back from Google to DDG/DeBERTa lazily load the existing NLI model through `verify.py`, which uses MPS when available and otherwise CPU.

The runner accepts either environment variable:

```bash
export GOOGLE_FACTCHECK_API_KEY="YOUR_KEY"
# or
export GOOGLE_FACT_CHECK_API_KEY="YOUR_KEY"
```

Run from the repository root:

```bash
python "google factchek/evaluation/eval_google_bucket.py"
```

Validate key configuration without importing the verifier or sending a request:

```bash
python "google factchek/evaluation/eval_google_bucket.py" --check-key-only
```

The fixed-bucket runner writes:

- `evaluation/google_escalation_results.csv` in this folder, with row ID, claim, true label, classifier label, committed DDG-only label, `google_candidate_count`, verification `source`, `verdict`, fallback `reason`, Google-specific `google_reason`, sanitized `google_error`, and hybrid `final_label`;
- `evaluation/google_escalation_summary.txt` in this folder, containing request/key-error and clean-no-match counts plus the summary metrics below.

The CSV is checkpointed after every row so an interrupted run preserves completed requests. The existing two-second inter-request delay is retained.

## Metric definitions

All rates use the fixed 100-row bucket unless a different denominator is stated.

- **Google coverage rate:** rows with at least one returned Google candidate divided by all rows.
- **Usable-verdict rate:** rows whose source is `google_fact_check` and whose verdict is `supported` or `refuted`, divided by all rows.
- **Google agreement with true labels:** usable Google verdicts whose mapped label matches `true_label`, divided by the number of usable Google verdicts. This is `n/a` when there are no usable verdicts.
- **Classifier-only accuracy:** rows where `classifier_label == true_label`, divided by all rows. The frozen bucket is validated as 76/100 before requests begin.
- **Committed DDG-only final accuracy:** rows where the committed `final_label == true_label`, divided by all rows. The frozen bucket is validated as 64/100 before requests begin.
- **Google-first hybrid accuracy:** rows where the new hybrid `final_label == true_label`, divided by all rows.

## Reproducibility boundary

The bucket, labels, and committed DDG-only outcomes are frozen repository artifacts. Google Fact Check and DDG are live external services: candidate availability, rankings, snippets, ratings, and fallback evidence can change between runs. The output CSV records one dated live run; it does not make retrieval responses reproducible. The API key must not be committed.
