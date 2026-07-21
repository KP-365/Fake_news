# Results and Discussion

**The classifier achieved near-ceiling performance within WELFake, but this result does not establish cross-dataset generalisation.** We evaluated RoBERTa-LoRA on the deterministic, stratified 70/15/15 split generated with seed 42. The held-out test set contained 9,398 unseen articles, comprising 5,193 real and 4,205 fake articles. We define accuracy as the proportion classified correctly, precision as $TP/(TP+FP)$, recall as $TP/(TP+FN)$, class F1 as $2PR/(P+R)$, and each macro score as the unweighted mean across the two classes.

| Model or class | Accuracy | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| RoBERTa-LoRA overall | 0.9957 | 0.9958 | 0.9956 | 0.9957 | 9,398 |
| Real | | 0.9952 | 0.9971 | 0.9962 | 5,193 |
| Fake | | 0.9964 | 0.9941 | 0.9952 | 4,205 |
| TF-IDF + logistic regression | 0.9526 | | | 0.9521 | |
| Majority-class baseline | 0.5526 | | | 0.3559 | |

**Table 1.** Held-out WELFake performance and baseline comparison.

The confusion matrix uses true labels as rows and predicted labels as columns in real/fake order.

```text
[[5178,   15],
 [  25, 4180]]
```

The classifier made 40 errors. It labelled 15 real articles as fake and 25 fake articles as real, which agrees with the lower fake-class recall of 0.9941 compared with the real-class recall of 0.9971. RoBERTa-LoRA exceeded TF-IDF by 4.31 percentage points in accuracy and 4.36 points in macro F1. This margin supports contextual representation learning over word-frequency features on this split, but it does not show the same advantage on LIAR or under distribution shift.

**MC Dropout improved discrimination without improving calibration.** Across 30 stochastic passes, the classifier reached 0.9967 accuracy compared with 0.9957 under deterministic inference. However, ECE was 0.0055 compared with 0.0028 for deterministic confidence. The higher ECE means that MC averaging improved accuracy without bringing confidence closer to observed accuracy.

**Automatic NLI override failed on the cases selected for intervention.** We used 30-pass MC Dropout to select the 100 lowest-confidence test articles, then applied DDG retrieval and DeBERTa NLI. The override mapped `supported` to real, `refuted` to fake, and retained the classifier label for `insufficient`.

| Verdict | Rows | Classifier accuracy | Overrides | Fixed | Broken | Final accuracy |
|---|---|---|---|---|---|---|
| Supported | 42 | 73.8% (31/42) | 14 | 7 | 7 | 73.8% (31/42) |
| Refuted | 32 | 71.9% (23/32) | 20 | 4 | 16 | 34.4% (11/32) |
| Insufficient | 26 | 84.6% (22/26) | 0 | 0 | 0 | 84.6% (22/26) |
| **Overall** | **100** | **76.0% (76/100)** | **34** | **11** | **23** | **64.0% (64/100)** |

**Table 2.** Escalation outcomes on the 100 lowest-confidence articles.

Escalation reduced accuracy from 76.0% to 64.0%, a loss of 12 percentage points. Across 34 label changes, it fixed 11 classifier mistakes but broke 23 correct predictions. The `supported` group was neutral with seven fixes and seven breaks. The `refuted` group caused the main failure, with four fixes and 16 breaks and a fall from 71.9% to 34.4%. The `insufficient` group retained its labels and achieved the best classifier-only accuracy of 84.6%. We therefore reject the current unthresholded override. A contradiction score from a live snippet does not establish that a complete article is false, so external evidence should support human review or a policy with validated evidence-quality thresholds.

**Google Fact Check supplied no coverage on the fixed bucket, while its live fallback was unstable.** The API produced 0/100 coverage. Every row recorded `no Google fact-check match`, with 100/100 clean no-match responses and 0/100 request or key errors. The DDG fallback verdict distribution shifted from 42/32/26 supported, refuted, and insufficient verdicts to 30/12/58 on the repeated run, while both runs retained 64.0% accuracy. The unchanged aggregate therefore concealed live-retrieval instability.

**The deployed pipeline worked, but its supporting evidence remained fallible.** We verified the public ZeroGPU Space on 20 July 2026 through its `/analyze` endpoint. The call completed in 64.4 seconds and the classifier returned REAL at 99.57% confidence, Very stable prediction stability, and an NLI verdict of Supported. This runtime included live DDG retrieval and two ZeroGPU allocations. In a separate single-reviewer assessment, 8 of 10 explanations matched their structured inputs, giving 80% descriptive faithfulness. The two failures were consequential. Row 8 reversed a `supported` NLI verdict, while row 94 attributed the classifier decision to retrieved evidence that the classifier never received.

These results have four boundaries. First, the classifier metrics cover one held-out WELFake split rather than LIAR or another external corpus. Second, escalation covers 100 of 9,398 test articles and therefore measures behaviour only in the lowest-confidence bucket. Third, live DDG results can change between runs. Fourth, the explanation review used 10 examples and one reviewer, so 80% is a sample description rather than a population estimate or multi-reviewer agreement measure.

# Conclusions

**The classifier performed strongly in-dataset, but automatic evidence override made the selected predictions worse.** RoBERTa-LoRA reached 0.9957 accuracy and 0.9957 macro F1 on 9,398 held-out WELFake articles and exceeded TF-IDF accuracy by 4.31 percentage points. MC Dropout raised accuracy to 0.9967 but increased ECE from 0.0028 to 0.0055, so it did not improve calibration. NLI override reduced low-confidence accuracy from 76.0% to 64.0% because it broke 23 correct predictions while fixing 11 errors. Google Fact Check then returned 0/100 coverage, while repeated DDG verdicts changed despite identical 64.0% accuracy. We therefore keep the classifier label final and present retrieved evidence as a separate signal for review. The operational Space and 8 of 10 explanation result show that the complete pipeline runs, while the two attribution failures show why evidence conflicts must remain visible. Cross-dataset testing, validated routing thresholds, stable evidence snapshots, and a larger multi-reviewer explanation study remain necessary before we claim broader reliability.

**Word count.** 768 words, excluding tables, headings, captions, equations, code blocks, and this note.

**Estimated column length.** Approximately two CVPR-style columns, including the two compact tables.

**Source check.** We cross-checked every quantitative result against the committed notebook outputs, MC arrays, escalation CSV, Google CSV and summary text, faithfulness-review CSV, and deployment evidence notes.
