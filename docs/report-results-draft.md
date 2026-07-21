# Results (Draft)

This draft renders outline subsections A.1, A.2, A.3, A.5, and A.6 from `docs/report-results-discussion-outline.md` in academic report prose. Every number is reproduced verbatim from that outline; no new figures are introduced. Subsections still awaiting committed evidence (A.4, A.7) are retained as clearly marked placeholders only.

## A.1 Experimental setup

All experiments were conducted on the WELFake corpus using a deterministic, stratified 70/15/15 train/validation/test split generated with seed 42, ensuring that the class distribution was preserved across partitions and that the split can be reproduced exactly. The held-out test set comprises 9,398 articles: 5,193 labelled real and 4,205 labelled fake. None of these articles was used for training, so all reported classifier metrics reflect performance on unseen data.

It should be emphasised that these are in-dataset results on the held-out WELFake split rather than a cross-dataset evaluation on LIAR or any external corpus; this scope is retained throughout the Results section and returned to in the Discussion.

All reported metrics are defined on the binary confusion counts. For a given class, $\mathrm{TP}$ denotes true positives (articles of that class predicted as that class), $\mathrm{FP}$ false positives (articles of the other class predicted as that class), $\mathrm{TN}$ true negatives (articles of the other class predicted as the other class), and $\mathrm{FN}$ false negatives (articles of that class predicted as the other class).

## A.2 Classifier performance

The RoBERTa-LoRA classifier was evaluated on the full held-out test set. Accuracy is the share of all test articles predicted correctly:

$$
\mathrm{Accuracy} = \frac{\mathrm{TP} + \mathrm{TN}}{\mathrm{TP} + \mathrm{FP} + \mathrm{TN} + \mathrm{FN}}
$$

where the counts are taken over the full test set. Its overall metrics are presented in Table 1.

| Metric | Value |
|---|---:|
| Accuracy | 0.9957 |
| Macro precision | 0.9958 |
| Macro recall | 0.9956 |
| Macro F1 | 0.9957 |

**Table 1.** Held-out WELFake metrics for the RoBERTa-LoRA classifier.

Per-class precision, recall, and F1 are defined for each class as

$$
\mathrm{Precision}_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FP}_c}, \qquad
\mathrm{Recall}_c = \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FN}_c}, \qquad
F1_c = \frac{2 \cdot \mathrm{Precision}_c \cdot \mathrm{Recall}_c}{\mathrm{Precision}_c + \mathrm{Recall}_c}
$$

where $\mathrm{TP}_c$, $\mathrm{FP}_c$, and $\mathrm{FN}_c$ are the true-positive, false-positive, and false-negative counts for class $c \in \{\text{real}, \text{fake}\}$. The macro metrics in Table 1 are the unweighted mean of the per-class scores, matching scikit-learn's `average="macro"` as used in `eval_FakeNews.ipynb`; for example, macro F1 is

$$
F1_{\mathrm{macro}} = \frac{1}{|C|} \sum_{c \in C} F1_c, \qquad C = \{\text{real}, \text{fake}\}.
$$

Per-class results are reported in Table 2.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Real | 0.9952 | 0.9971 | 0.9962 | 5,193 |
| Fake | 0.9964 | 0.9941 | 0.9952 | 4,205 |

**Table 2.** Per-class held-out performance, with support counts.

The confusion matrix, with true labels as rows and predicted labels as columns in real/fake order, is:

```text
[[5178,   15],
 [  25, 4180]]
```

The classifier made 40 errors in total: 15 real articles were labelled fake and 25 fake articles were labelled real. The slightly higher count of false-real errors is consistent with the marginally lower fake-class recall (0.9941) relative to real-class recall (0.9971). The corresponding confusion-matrix figure is preserved in `eval_FakeNews.ipynb`.

## A.3 Baseline comparison

The classifier was compared against two baselines on the same held-out split: a TF-IDF logistic regression model and a majority-class baseline, which always predicts the most common training label and serves as a lower-bound reference. Results are given in Table 3.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| RoBERTa-LoRA | 0.9957 | 0.9957 |
| TF-IDF + logistic regression | 0.9526 | 0.9521 |
| Majority-class baseline | 0.5526 | 0.3559 |

**Table 3.** Held-out WELFake comparison against baselines.

RoBERTa-LoRA exceeds the TF-IDF baseline by 4.31 percentage points in accuracy and 4.36 points in macro F1, indicating that the fine-tuned transformer captures patterns beyond the word-frequency features available to the linear model. The macro-F1 comparison is also presented as a bar chart preserved in `eval_FakeNews.ipynb`.

One reporting caveat applies: these baseline values are taken from the notebook's preserved output, since no dedicated baseline CSV is committed under `evaluation/`; the only committed evaluation CSV is the escalation results file discussed below.

## A.4 MC Dropout calibration

> **TODO - placeholder pending Kayleb's notebook run.** The committed `eval_MCFakeNews.ipynb` implements the 30-pass MC Dropout evaluation, predictive-entropy and mutual-information diagnostics, a rejection curve, a 15-bin reliability diagram, and ECE calculations for MC-averaged and deterministic confidence, but the notebook's executed outputs are not committed. No numerical calibration, ECE, entropy, or accuracy-versus-coverage result is available to quote. This subsection will be completed once the notebook is fully executed and its outputs are committed.

## A.5 Escalation experiment

To test whether external evidence could rescue the classifier's most uncertain predictions, the 100 WELFake test articles with the lowest MC Dropout confidence were escalated to a verification layer combining DDG retrieval with DeBERTa NLI. The layer produced 42 supported, 32 refuted, and 26 insufficient verdicts. The final label is determined by the override rule implemented in `evaluation/eval_escalation.py` (`VERDICT_TO_LABEL` with a fallback to the classifier label):

$$
\hat{y}^{\mathrm{final}}_i =
\begin{cases}
\text{real}, & v_i = \text{supported} \\
\text{fake}, & v_i = \text{refuted} \\
\hat{y}^{\mathrm{clf}}_i, & v_i = \text{insufficient}
\end{cases}
$$

where $v_i$ is the NLI verdict for article $i$ and $\hat{y}^{\mathrm{clf}}_i$ is the classifier's label for that article.

For each verdict group $g$ of size $n_g$, classifier accuracy before escalation and final accuracy after escalation are the ratios

$$
\mathrm{Acc}^{\mathrm{clf}}_g = \frac{1}{n_g} \sum_{i \in g} \mathbb{1}\!\left[\hat{y}^{\mathrm{clf}}_i = y_i\right], \qquad
\mathrm{Acc}^{\mathrm{final}}_g = \frac{1}{n_g} \sum_{i \in g} \mathbb{1}\!\left[\hat{y}^{\mathrm{final}}_i = y_i\right]
$$

where $y_i$ is the true label of article $i$ and $\mathbb{1}[\cdot]$ is the indicator function, equal to 1 when its condition holds and 0 otherwise; the overall row uses $g$ equal to the full 100-article bucket. The results, recorded in `evaluation/escalation_results.csv`, are presented in Table 4.

| Verdict | Rows | Classifier accuracy | Actual overrides | Fixed | Broken | Final accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Supported | 42 | 73.8% (31/42) | 14 | 7 | 7 | 73.8% (31/42) |
| Refuted | 32 | 71.9% (23/32) | 20 | 4 | 16 | 34.4% (11/32) |
| Insufficient | 26 | 84.6% (22/26) | 0 | 0 | 0 | 84.6% (22/26) |
| **Overall** | **100** | **76.0% (76/100)** | **34** | **11** | **23** | **64.0% (64/100)** |

**Table 4.** Escalation outcomes on the 100 lowest-confidence test articles. "Fixed" denotes classifier errors corrected by the override; "broken" denotes originally correct predictions inverted by it.

Escalation reduced accuracy on this low-confidence bucket from 76.0% to 64.0%, a loss of 12 percentage points. Across the 34 label changes, the override fixed 11 classifier mistakes but broke 23 predictions that were originally correct. The supported verdicts were neutral overall, with seven fixes and seven breaks, whereas the refuted verdicts were the dominant failure mode: four fixes against 16 breaks, reducing that group's accuracy from 71.9% to 34.4%. The insufficient group, whose labels were correctly left unchanged, had the highest classifier-only accuracy at 84.6%.

The committed CSV also shows that the sample contained 66 true-real and 34 true-fake rows, against 62 classifier-real and 38 classifier-fake predictions before escalation, and 56 real and 44 fake final labels after it. It must be stressed that this experiment covers only the 100 least-confident test articles, not the full 9,398, and therefore characterises the verification layer's behaviour precisely where the classifier is least certain rather than across the whole distribution.

## A.6 Deployed Space verification

The complete system was deployed publicly as a ZeroGPU Hugging Face Space at https://wf1212-fake-news-detector.hf.space. End-to-end operation was verified on 20 July 2026 with a `gradio_client` call to the Space's named `/analyze` endpoint using the headline "Federal Reserve holds interest rates steady amid mixed economic data".

The call completed in 64.4 seconds wall-clock and returned a final classifier label of REAL at 99.57% confidence, with prediction stability rated Very stable and an NLI verdict of Supported. The measured runtime includes live DDG retrieval and two ZeroGPU allocations within the request. Two reporting caveats apply: no permanent transcript of the full response payload is committed, so only these documented summary values are reported; and the repository contains no committed runtime-status artifact, so the deployment claim rests on the documented endpoint verification rather than a persisted state record.

## A.7 Explanation faithfulness

> **TODO - placeholder pending the 10-row human review.** `eval_faithfulness.ipynb` provides the manual review workflow, sampling 10 escalation rows with `np.random.default_rng(42)`, regenerating explanations from structured signals, and exporting `evaluation/faithfulness_review.csv`. No completed review CSV is committed, and no explanation-faithfulness or human-evaluation result has been completed. No faithfulness rate is reported here until the review is committed.
