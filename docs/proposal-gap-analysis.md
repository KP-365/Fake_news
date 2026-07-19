# Proposal Gap Analysis

## Scope

This review compares the commitments in `proposal-group10-preview.pdf` with the repository state represented by `TASKS.md`, `scaffold_FakeNews_finn's_training.ipynb`, `eval_FakeNews.ipynb`, and `predict.py`.

Status meanings:

- **Done** — implemented and evidenced in the reviewed files.
- **Partial** — some implementation exists, but the stated commitment is incomplete.
- **Missing** — no implementation evidence exists in the reviewed files.
- **Deviated** — implementation materially differs from the proposal.

## Material deviations

1. **Dataset deviation — LIAR to WELFake:** the proposal makes LIAR the primary dataset and requires explicit targets on the held-out LIAR test set, with WELFake only as a second/backup source. The training and validation paths instead preprocess, split, train, and evaluate on WELFake. LIAR is downloaded and displayed but is not used for model training or reported evaluation.
2. **Model deviation — BERT to RoBERTa:** the proposal specifies `bert-base-uncased`. The implemented classifier and inference loader use `roberta-base`.

## Commitment-by-commitment review

| # | Proposal commitment | Status | Evidence |
|---:|---|---|---|
| 1 | Deliver a three-layer fake-news detection, verification, and explanation framework. | **Partial** | Layer 1 classification exists; the training notebook's Layer 2 and Layer 3 implementation cells are empty, and `TASKS.md` leaves all integration items unchecked. |
| 2 | Classify an article or claim as **real or fake**. | **Done** | `predict.py` tokenizes one text, runs the checkpoint, and returns `ID_TO_LABEL = {0: "real", 1: "fake"}` with confidence. |
| 3 | Use linguistic features learned from labelled datasets. | **Done** | The training notebook cleans and combines WELFake titles/text, tokenizes the combined content, and fine-tunes a sequence classifier. |
| 4 | Use a **BERT-based** classifier. | **Deviated** | Both training model initializations and `predict.py::BASE_MODEL_NAME` use `roberta-base`, not `bert-base-uncased`. |
| 5 | Fine-tune with **LoRA**, keeping most base weights frozen. | **Done** | The training notebook applies PEFT `LoraConfig` to query/value modules and records 294,912 trainable parameters; `predict.py` reloads the saved PEFT adapter. |
| 6 | Use Monte Carlo Dropout for prediction uncertainty. | **Missing** | The notebook's “Monte Carlo Dropout” section contains only a classifier-head save command; there are no repeated stochastic inference passes. |
| 7 | Allow an **uncertain** outcome rather than always forcing a binary answer. | **Missing** | `predict.py` always takes `argmax` and returns `real` or `fake`; it defines no uncertainty or deferral threshold. |
| 8 | Integrate the Google Fact Check Tools API. | **Missing** | The notebook's fact-check verification code cell is empty and `TASKS.md` leaves the API integration unchecked. |
| 9 | Search verified sources such as PolitiFact and Snopes for supporting or refuting evidence. | **Missing** | No fact-check request, source query, or evidence structure appears in the reviewed implementation files. |
| 10 | Detect classifier/fact-check disagreements. | **Missing** | The conflict-detection notebook cell is empty and the corresponding `TASKS.md` item is unchecked. |
| 11 | Analyse disagreement cases as difficult examples. | **Missing** | No conflict cases are produced, sampled, or analysed; `TASKS.md` leaves this task unchecked. |
| 12 | Generate a human-readable explanation of the final determination. | **Missing** | The explanation-agent notebook cell is empty; `predict.py` prints only a label and confidence. |
| 13 | Ground explanations in linguistic, verification, and uncertainty signals. | **Missing** | Verification and uncertainty outputs do not exist, and no explanation-composition code is present. |
| 14 | Use the Claude API (`claude-sonnet-4-6`) as the explanation agent. | **Missing** | `anthropic` is listed as a dependency/task resource, but there is no Claude client or model invocation in the reviewed files. |
| 15 | Make **LIAR** the primary dataset. | **Deviated** | The notebook downloads LIAR and prints its splits but trains `WelfakeDataset` from `train_df`, `valid_df`, and `test_df` created from WELFake. |
| 16 | Use **WELFake** as a backup/second source. | **Deviated** | WELFake became the sole training and reported evaluation source rather than a secondary source. |
| 17 | Evaluate explicit layer targets on the held-out **LIAR** test set. | **Deviated** | The training notebook and `eval_FakeNews.ipynb` evaluate the seeded WELFake test split; no LIAR test metrics are reported. |
| 18 | Build shared LIAR/WELFake preprocessing and tokenisation. | **Partial** | WELFake has cleaning, deduplication, splitting, and RoBERTa tokenisation; LIAR is loaded but is not transformed into the same binary classifier pipeline. |
| 19 | Report per-class precision and recall. | **Missing** | Neither evaluation notebook computes `precision_score`, `recall_score`, or a classification report. |
| 20 | Report **macro-averaged F1**. | **Missing** | Both notebooks call binary `f1_score` for label 1; `eval_FakeNews.ipynb` explicitly reports “F1 (fake class),” not macro-F1. |
| 21 | Report a 2 × 2 confusion matrix. | **Done** | `eval_FakeNews.ipynb` computes and plots `confusion_matrix(..., labels=[0, 1])`; its verified run produced `[[5178, 15], [25, 4180]]`. |
| 22 | Achieve macro-F1 competitive with published LIAR baselines. | **Missing** | There is no LIAR macro-F1 result or published-baseline comparison; the only reported F1 is WELFake fake-class F1. |
| 23 | Produce a reliability diagram. | **Missing** | The calibration/evaluation cell in the training notebook is empty and the validation notebook contains no reliability plot. |
| 24 | Compute Expected Calibration Error (ECE). | **Missing** | No calibration bins or ECE calculation appear in the reviewed files. |
| 25 | Show lower ECE than an uncalibrated softmax baseline. | **Missing** | `predict.py` exposes raw softmax confidence, with no calibration method or baseline comparison. |
| 26 | Report calibrated confidence in the final system. | **Partial** | `predict.py` reports maximum softmax probability as confidence, but no calibration has been implemented or demonstrated. |
| 27 | Evaluate selective prediction via accuracy on retained predictions as low-confidence cases are deferred. | **Missing** | There is no retention/coverage curve, confidence threshold, or deferral logic. |
| 28 | Measure fact-check verification coverage of test claims. | **Missing** | No verification layer is implemented, so coverage is not computed. |
| 29 | Measure classifier/fact-check disagreement frequency. | **Missing** | No retrieved verdicts or disagreement metric exist. |
| 30 | Review explanation faithfulness on a sample. | **Missing** | No explanations exist to review; `TASKS.md` leaves the faithfulness check unchecked. |
| 31 | Provide baseline models and evaluation plots. | **Missing** | The baseline notebook cell is empty and `TASKS.md` leaves baseline models/plots unchecked. |
| 32 | Deploy a Gradio interface. | **Missing** | The Gradio notebook cell is empty and no UI implementation is evidenced in the reviewed files. |
| 33 | Deploy publicly on Hugging Face Spaces. | **Missing** | No Space configuration or public deployment URL is recorded; `TASKS.md` leaves deployment unchecked. |
| 34 | End-to-end demo: classification, calibrated confidence, conflicting fact-checks, and readable explanation. | **Missing** | Only classification plus uncalibrated softmax confidence is available; the notebook end-to-end cell is empty. |
| 35 | Train on Google Colab using a free T4 GPU. | **Done** | The training notebook declares Colab/T4 metadata and contains completed three-epoch training outputs. |
| 36 | Use Python 3.10+ and the proposed ML libraries. | **Done** | Notebook metadata specifies Python 3.10, and the implementation imports Transformers, PEFT, Torch, pandas, KaggleHub, NumPy, and scikit-learn. |
| 37 | Use GitHub for version control, initially described as a private repository. | **Deviated** | GitHub version control is in use, but the evaluation notebook clones the repository anonymously and links to a public GitHub/Colab URL. |
| 38 | Supply a standalone way to run the trained classifier. | **Done** | `predict.py` loads the tracked adapter, tokenizer, and classifier head, then accepts a CLI text argument and prints label/confidence. |
| 39 | Preserve and load a deployable trained checkpoint. | **Done** | `predict.py` loads `models/roberta-trained-welfake`; the evaluation notebook loads it once before batched inference. |
| 40 | Evaluate on a held-out split rather than training data. | **Done** | Both notebooks recreate a stratified 70/15/15 WELFake split with `random_state=42` and reserve the final 15% for testing. |
| 41 | Report classifier results figures. | **Partial** | A confusion-matrix figure exists in `eval_FakeNews.ipynb`, but calibration, baseline-comparison, and LIAR result figures are absent. |
| 42 | Write Methodology and Implementation sections. | **Missing** | `TASKS.md` leaves these writing tasks unchecked, and none of the reviewed implementation files contains report sections. |
| 43 | Write Related Work, Results and Discussion, and Conclusions. | **Missing** | `TASKS.md` leaves these writing tasks unchecked; the reviewed files provide code/notebook guidance rather than report prose. |
| 44 | Write Introduction, Abstract, Evaluation, Contributions, and complete formatting. | **Missing** | `TASKS.md` leaves the writing/formatting bundle unchecked; no completed final report is evidenced in the reviewed files. |
| 45 | Produce presentation slides. | **Missing** | `TASKS.md` leaves presentation slides unchecked and the reviewed files contain no slide deliverable. |
| 46 | Assemble the final report and submission checklist. | **Missing** | `TASKS.md` leaves final report assembly/submission unchecked. |
| 47 | Have all group members contribute equally to experimental and writing work. | **Partial** | `TASKS.md` assigns work across K/E/W, but every checkbox remains unchecked and the reviewed files do not evidence completed contributions from all three owners. |

## Current validated result

The implemented path is a **RoBERTa-LoRA classifier trained and evaluated on WELFake**, not the proposed BERT-LoRA classifier evaluated primarily on LIAR. `eval_FakeNews.ipynb` currently reports 0.9957 accuracy and 0.9952 fake-class F1 on its recreated WELFake held-out split, plus a confusion matrix. These are strong in-dataset classification results, but they do not satisfy the proposal's LIAR, macro-F1, calibration, selective-deferral, verification, explanation, baseline, or deployment commitments.
