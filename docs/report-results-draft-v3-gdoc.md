# Results and Discussion

**The classifier reached near-ceiling accuracy within WELFake [Missing bibliography entry: verma2021welfake], but we do not treat that score as cross-dataset evidence.** We evaluated RoBERTa-LoRA on the deterministic, stratified 70/15/15 split with seed 42 rather than drawing a fresh random split, preserving class balance and reproducibility. The held-out test set contained 9,398 unseen articles, comprising 5,193 real and 4,205 fake articles. We report accuracy, per-class precision, recall and F1, with unweighted macro averages. Table 1 gives the scores that the error analysis then explains.

| Model or class | Accuracy | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| RoBERTa-LoRA overall | 0.9957 | 0.9958 | 0.9956 | 0.9957 | 9,398 |
| Real | | 0.9952 | 0.9971 | 0.9962 | 5,193 |
| Fake | | 0.9964 | 0.9941 | 0.9952 | 4,205 |
| TF-IDF + logistic regression | 0.9526 | | | 0.9521 | |
| Majority-class baseline | 0.5526 | | | 0.3559 | |

**Table 1.** Held-out WELFake performance and baseline comparison.

**The 40 errors show why near-ceiling performance still needs a dataset boundary.** We use true labels as rows and predicted labels as columns in real/fake order. The matrix below exposes that error pattern.

| | Predicted real | Predicted fake |
|---|---:|---:|
| True real | 5178 | 15 |
| True fake | 25 | 4180 |

The classifier labelled 15 real articles as fake and 25 fake articles as real. This pattern agrees with the lower fake-class recall of 0.9941 compared with the real-class recall of 0.9971. RoBERTa-LoRA exceeded TF-IDF by 4.31 percentage points in accuracy and 4.36 points in macro F1. This margin supports contextual representations over word-frequency features on this split, but it does not establish the same advantage on LIAR or under distribution shift. Near-ceiling WELFake accuracy may reflect source or style shortcut learning rather than transferable veracity cues, a known failure mode in which a model performs well on a benchmark but fails under harder transfer conditions (Geirhos et al., 2020). We next test whether MC Dropout identifies the few cases that remain difficult.

**MC Dropout left accuracy effectively unchanged, while entropy separation supplied the operative result.** Across 30 stochastic passes, MC Dropout [Missing bibliography entry: gal2016dropout] reached 0.9967 accuracy compared with 0.9957 under deterministic inference, a difference of 0.10 percentage points. Its ECE [Missing bibliography entry: guo2017calibration] was 0.0055 compared with 0.0028 for deterministic confidence, so MC averaging did not improve calibration. Mean predictive entropy was 0.0333 for correct predictions and 0.4816 for incorrect predictions. On the notebook's rejection curve under MC labels, deferring the 100 highest-entropy cases left 9,298 articles at 98.94% coverage and raised retained accuracy to 99.88% (9,287/9,298). The useful finding is therefore error separation for routing, not a material accuracy gain or better calibration. That routing result leads to the test of whether verification can rescue the deferred cases.

**We expected external evidence to rescue uncertain predictions, but direct NLI override produced the opposite result.** We selected the 100 lowest-confidence test articles rather than a random 100 because the experiment targeted the cases identified for intervention, not average behaviour across all 9,398 articles [Missing bibliography entry: geifman2017selective]. We tested direct override rather than the deployed advisory use of NLI so that the evaluation could measure whether external evidence corrected classifier errors. The rule mapped `supported` to real and `refuted` to fake, while `insufficient` retained the classifier label. We then applied DDG retrieval and DeBERTa NLI. Table 2 records how that expectation failed.

| Verdict | Rows | Classifier accuracy | Overrides | Fixed | Broken | Final accuracy |
|---|---|---|---|---|---|---|
| Supported | 42 | 73.8% (31/42) | 14 | 7 | 7 | 73.8% (31/42) |
| Refuted | 32 | 71.9% (23/32) | 20 | 4 | 16 | 34.4% (11/32) |
| Insufficient | 26 | 84.6% (22/26) | 0 | 0 | 0 | 84.6% (22/26) |
| **Overall** | **100** | **76.0% (76/100)** | **34** | **11** | **23** | **64.0% (64/100)** |

**Table 2.** Escalation outcomes on the 100 lowest-confidence articles.

**The override moved accuracy in a consistently harmful direction, although this sample does not establish a statistically significant difference.** Baseline classifier accuracy was 76.0% (76/100; 95% Wilson CI, 66.8% to 83.3%). Post-override accuracy was 64.0% (64/100; 95% Wilson CI, 54.2% to 72.7%). Overrides fixed 11 errors and broke 23 correct predictions; the two-sided exact McNemar test gave p = 0.058, a direction consistent with harm that does not reach the conventional 0.05 threshold in this paired sample of n = 100. The `supported` group was neutral with seven fixes and seven breaks. The `refuted` group caused the main failure, with four fixes and 16 breaks and a fall from 71.9% to 34.4%. The `insufficient` group retained its labels and achieved the best classifier-only accuracy of 84.6%. We therefore reject the current unthresholded override. A contradiction score from a live snippet does not establish that a complete article is false, so external evidence should support human review or a policy with validated evidence-quality thresholds. This failure makes retrieval coverage and source stability the next question.

**Google Fact Check supplied no coverage on the fixed bucket, while its live fallback was unstable.** The API produced 0/100 coverage. Every row recorded `no Google fact-check match`, with 100/100 clean no-match responses and 0/100 request or key errors. This result is consistent with a task mismatch because Google Fact Check uses claim-level indexing, whereas our unit of evaluation was a full article. The DDG fallback verdict distribution shifted from 42/32/26 supported, refuted, and insufficient verdicts to 30/12/58 on the repeated run, while both runs retained 64.0% accuracy. The unchanged aggregate concealed live-retrieval instability. Coverage alone does not show whether the deployed output remained faithful to its inputs, which we test next.

**The deployed pipeline ran, but its explanation layer remained fallible.** We verified the public ZeroGPU Space on 20 July 2026 through its `/analyze` endpoint. The call completed in 64.4 seconds and the classifier returned REAL at 99.57% confidence, "Very stable" prediction stability, and an NLI verdict of Supported. This runtime included live DDG retrieval and two ZeroGPU allocations. Explanation faithfulness was 80.0% (8/10; 95% Wilson CI, 49.0% to 94.3%). This was a single-reviewer assessment rather than a population estimate. Row 8 reversed a `supported` NLI verdict, while row 94 attributed the classifier decision to retrieved evidence that the classifier never received. These operating and faithfulness results complete the evidence needed to judge the proposal criteria.

**Against the proposal's five success criteria, the system met four.** **MET, classifier quality.** Macro F1 reached 0.9957 on held-out WELFake [Missing bibliography entry: verma2021welfake], which met the proposal's published-baseline target but did not establish LIAR performance. **NOT MET, calibration.** MC ECE was 0.0055 rather than below deterministic ECE of 0.0028 [Missing bibliography entry: guo2017calibration]. **MET, selective deferral.** Retained accuracy rose when we deferred the 100 highest-entropy predictions [Missing bibliography entry: geifman2017selective]. **MET, verification measurement.** Table 2 records coverage and classifier-verifier disagreement, while Google Fact Check returned 0/100 coverage. **MET, explanation faithfulness.** Eight of 10 sampled explanations matched their structured inputs. The unmet calibration criterion and the evidence-quality failures lead directly to the study limits.

**Four limits bound these results.** First, the classifier metrics cover one held-out WELFake split rather than LIAR or another external corpus. Second, escalation covers 100 of 9,398 test articles and therefore measures behaviour only in the lowest-confidence bucket. Third, live DDG results can change between runs. Fourth, the explanation review used 10 examples and one reviewer, which leaves a wide 49.0% to 94.3% Wilson interval and provides no multi-reviewer agreement measure. These boundaries constrain the conclusion and define the next evaluation steps.

# Conclusions

**The classifier met its in-dataset target, but the evidence layer did not earn authority over its labels.** We built and publicly deployed the proposal's three-layer detection, verification, and explanation system using RoBERTa-LoRA on WELFake with DDG and DeBERTa NLI. MC Dropout supplied a useful routing signal through entropy separation, not a material accuracy or calibration gain. The escalation experiment then found a consistent direction of harm from direct override, although the sample was too small to cross the 0.05 significance threshold. This result extends FEVER-style retrieval and entailment work by showing that a live verifier can reduce end-to-end accuracy when its verdict directly replaces an article label (Thorne et al., 2018; Hanselowski et al., 2018; Soleimani et al., 2020). Google Fact Check's claim-level index did not cover the article bucket, and repeated DDG retrieval changed verdicts. We therefore keep the classifier label final and present retrieved evidence as a separate signal for review. Cross-dataset testing, validated routing thresholds, stable evidence snapshots, and a larger multi-reviewer explanation study are the next tests of whether this system generalises.
