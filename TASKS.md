# Task Tracker

Live status lives in [GitHub Issues](../../issues) and the
[Project board](../../projects) — this file is the static map from the proposal's
Contributions section to concrete, checkable tasks. Tick items off as issues close.

**Deadline: 21 July 2026** (tentative — confirm against QMplus/module page).

Legend: **K** = Kayleb Parkes, **E** = Eric Kamalendran, **W** = William McKie

## Milestone 0 — Setup (due 10 Jul)

- [ ] Repo scaffold, `requirements.txt`, environment (Colab T4) — K
- [ ] Acquire & load datasets: LIAR (primary), WELFake (backup) — E

## Milestone 1 — Layer 1: Classification (owner: K, due 15 Jul)

- [ ] BERT-base-uncased + LoRA fine-tuning (Hugging Face PEFT)
- [ ] Monte Carlo Dropout for uncertainty estimation
- [ ] Per-class precision/recall, macro-F1, confusion matrix on LIAR test set
- [ ] Reliability diagram + Expected Calibration Error (ECE) vs. uncalibrated softmax baseline
- [ ] Selective prediction: accuracy-on-retained as least-confident cases are deferred
- [ ] Writing: Methodology, Implementation, classifier results figures

## Milestone 2 — Layer 2: Verification (owner: E, due 15 Jul)

- [ ] Data preprocessing & tokenisation pipeline (LIAR/WELFake)
- [ ] Google Fact Check Tools API integration (PolitiFact, Snopes)
- [ ] Conflict detection: classifier vs. retrieved fact-check disagreement
- [ ] Analysis of conflict cases as hardest examples; verification layer coverage metric
- [ ] Writing: Related Work, Results and Discussion, Conclusions

## Milestone 3 — Layer 3: Explanation, Baselines & UI (owner: W, due 18 Jul)

- [ ] Baseline models + evaluation plots (comparison to BERT-LoRA)
- [ ] Explanation agent (combines linguistic + verification + uncertainty signals)
- [ ] Gradio UI + Hugging Face Spaces deployment (public URL)
- [ ] Presentation slides
- [ ] Writing: Introduction, Abstract, Evaluation, Contributions, formatting

## Milestone 4 — Integration & Evaluation (team, due 21 Jul — submission)

- [ ] Faithfulness check: sample explanations reviewed against underlying signals
- [ ] End-to-end demo test (classify → confidence → conflicts → explanation)
- [ ] Final report assembly and submission checklist

## Success criteria (from proposal)

The project succeeds when the deployed Gradio demo classifies a claim, reports a
calibrated confidence, surfaces any conflicting fact-checks, and returns a readable
explanation — with each layer meeting its explicit target on the LIAR test set (WELFake
as second source).
