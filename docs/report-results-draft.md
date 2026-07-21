# Results and Discussion (Draft)

This draft renders outline subsections A.1 through A.7 and B.1 through B.6 from `docs/report-results-discussion-outline.md` in academic report prose. Every reported result is reproduced from a committed artifact; no new figures are introduced.

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

The merged `eval_MCFakeNews.ipynb` (commit `799f359`) preserves an executed uncertainty evaluation with 30 stochastic forward passes per test article. Let $p_{tic}$ denote the softmax probability assigned to class $c$ for article $i$ during stochastic pass $t$, and let $T=30$ denote the number of passes. The probability used for MC prediction is the pass average

$$
\bar{p}_{ic} = \frac{1}{T}\sum_{t=1}^{T}p_{tic}.
$$

Here, $\bar{p}_{ic}$ is the MC-averaged probability for article $i$ and class $c$. Labels are selected by the maximum component of $\bar{p}_{i}$, matching the notebook's `mean_probabilities.argmax(axis=1)` implementation.

MC averaging produced accuracy 0.9967 and macro F1 0.9967, compared with deterministic accuracy 0.9957 and macro F1 0.9957. The MC confusion matrix, with true labels as rows and real/fake order, is

```text
[[5175,   18],
 [  13, 4192]]
```

The notebook also preserves the corresponding embedded MC confusion-matrix figure.

Uncertainty was decomposed using predictive entropy and mutual information. Let $C$ denote the two-class label set and let $\epsilon>0$ denote the small numerical stabiliser added before taking logarithms. Predictive entropy for article $i$ is

$$
H_i = -\sum_{c \in C}\bar{p}_{ic}\log(\bar{p}_{ic}+\epsilon).
$$

The mean entropy of the per-pass predictions is

$$
\bar{H}^{\mathrm{pass}}_i = -\frac{1}{T}\sum_{t=1}^{T}\sum_{c \in C}p_{tic}\log(p_{tic}+\epsilon),
$$

and mutual information, used as the epistemic uncertainty estimate, is

$$
\mathrm{MI}_i = H_i - \bar{H}^{\mathrm{pass}}_i.
$$

The executed outputs report mean predictive entropy of 0.0347 over all test articles. Correct predictions had mean predictive entropy 0.0333, whereas incorrect predictions had 0.4816. Mean mutual information was 0.0015 overall, 0.0013 for correct predictions, and 0.0468 for incorrect predictions. Thus both reported uncertainty measures were substantially higher among errors. The notebook's embedded uncertainty figure presents the correct-versus-incorrect predictive-entropy boxplot alongside a rejection curve that orders articles from most to least certain.

Calibration was measured with 15 equal-width confidence bins. Let $N$ be the number of test articles; let $q_i=\max_{c\in C}\bar{p}_{ic}$ be article $i$'s confidence; let $y_i$ be its true label; let $\mathbb{1}[\cdot]$ denote the indicator function; let $z_i=\mathbb{1}[\arg\max_{c\in C}\bar{p}_{ic}=y_i]$ indicate whether its prediction is correct; and let $B_b$ contain the articles assigned to confidence bin $b$. Bin accuracy and confidence are

$$
\mathrm{acc}(B_b)=\frac{1}{|B_b|}\sum_{i\in B_b}z_i, \qquad
\mathrm{conf}(B_b)=\frac{1}{|B_b|}\sum_{i\in B_b}q_i.
$$

The notebook implements expected calibration error as

$$
\mathrm{ECE}=\sum_{b=1}^{15}\frac{|B_b|}{N}\left|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)\right|.
$$

Here, $|B_b|$ is the number of articles in bin $b$, and the other symbols are defined above. The executed ECE was 0.0055 for MC Dropout and 0.0028 for deterministic confidence. Since a lower ECE indicates closer agreement between confidence and observed accuracy, MC averaging did not improve calibration in this run, despite its higher accuracy and macro F1. The notebook's embedded 15-bin reliability diagram visualises this comparison directly.

These results are backed by the notebook's executed outputs, embedded figures, and four arrays now committed under `evaluation/mc_arrays/`. Independently loading the arrays reproduces the reported predictive-entropy mean, mutual-information mean, argmax accuracy, and confusion matrix.

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

The completed human review in commit `e529b46` assessed 10 generated explanations against their supplied structured signals. Eight explanations were marked `yes` for matching those signals and two were marked `no`, giving a faithfulness rate of 8/10 (80%) (`evaluation/faithfulness_review.csv`).

Both failures reveal concrete attribution errors. For source row 8, the NLI verdict was `supported`, with entailment 0.98, and therefore conflicted with the fake classifier label. The explanation instead claimed that the evidence aligned with and reinforced the fake label, reversing the direction of the NLI verdict. For source row 94, the explanation attributed the classifier's decision to five retrieved evidence items even though those items were not inputs to the classifier.

The 80% rate should be treated as descriptive rather than definitive: it comes from a small 10-explanation sample assessed by a single reviewer. A larger review with multiple independent raters is required to estimate broader explanation faithfulness and inter-rater reliability.

# Discussion

## B.1 Interpreting near-ceiling in-dataset accuracy

The RoBERTa-LoRA classifier achieved 0.9957 accuracy and 0.9957 macro F1 on the held-out WELFake test set, placing its in-dataset performance close to the metric ceiling. The confusion profile provides necessary context for this aggregate result. The classifier made 40 errors, comprising 15 false-fake predictions and 25 false-real predictions. Thus, the high overall score does not imply error-free classification, and the remaining errors occur in both directions (`docs/results-summary.md` §1).

The comparison with TF-IDF further indicates that the transformer contributes beyond word-frequency patterns. Its accuracy was 4.31 percentage points higher than the TF-IDF baseline on the same held-out split. This margin supports the value of contextual representation learning within WELFake, but it does not establish equivalent gains under distribution shift (`docs/results-summary.md` §1).

The scope of this finding is therefore narrow. All classifier results are in-dataset measurements from one corpus, not LIAR results or a cross-dataset evaluation. WELFake's cleaned, label-verified construction may contribute to the near-ceiling performance, but this is an interpretation rather than a measured explanation because no committed cross-dataset artifact tests it (`docs/results-summary.md` §4; `docs/writeup-facts-w.md` §5).

## B.2 Why NLI override failed

The escalation experiment produced a clear negative result. On the low-confidence bucket, accuracy fell from 76.0% to 64.0% across 34 label changes. The override corrected 11 classifier mistakes but broke 23 predictions that had originally been correct. The imbalance between fixes and breaks shows that the evidence layer did not merely add noise around an unchanged result; its automatic decisions systematically reduced accuracy on the cases selected for intervention (`docs/results-summary.md` §3).

The `refuted` group was the principal failure mode. Its override produced 4 fixes and 16 breaks, reducing accuracy within that group from 71.9% to 34.4%. By contrast, the `supported` group was neutral, with 7 fixes and 7 breaks. The `insufficient` group retained the classifier label and had the highest classifier accuracy at 84.6%. These outcomes indicate that abstaining was safer than forcing a label when the verifier's evidence was not strong enough to resolve the claim (`docs/results-summary.md` §3).

The result is consistent with limitations in the evidence-to-label mapping. Retrieved snippets may be irrelevant, ambiguous, or concerned with a related claim, while a contradiction score does not by itself establish that the full article is false. Mapping every `refuted` verdict directly to fake therefore treats the NLI relation as a reliable article-level judgement without first validating evidence relevance or the decision threshold. The experiment shows that this unthresholded override policy is unsafe, not that retrieval or NLI has no value. The evidence is better suited to contextual display, human review, or a future policy with evidence-quality checks and validated thresholds (`docs/results-summary.md` §3).

The explanation review exposes the same evidence-conflict weakness in an independent component. In the escalation rule, `refuted` evidence displaced classifier labels and broke correct predictions. In source row 8 of the faithfulness review, a `supported` NLI verdict that conflicted with a fake classifier label was instead narrated as evidence reinforcing fake. The decision layer and explanation layer perform different functions, yet both failed to preserve the direction and role of an evidence verdict when it conflicted with the classifier output. This parallel does not demonstrate a shared implementation defect, but it does show a common system-level risk: evidence can be made to control or rationalise a label rather than being represented as a separate, potentially conflicting signal (`evaluation/escalation_results.csv`; `evaluation/faithfulness_review.csv`, commit `e529b46`).

This risk is amplified by the use of live DDG data, because search results can change between runs. The verifier's output is therefore sensitive both to the semantics of retrieved snippets and to their availability at query time. These findings support the deployed policy in which the classifier remains final and NLI evidence is presented only as context (`docs/results-summary.md` §4).

## B.3 Google Fact Check coverage and live-retrieval instability

The Google-first experiment in commit `034f8bf` evaluated the same frozen 100-row low-confidence bucket as the original escalation experiment. The Google Fact Check API returned zero candidates, giving 0.0% coverage (0/100) and no usable verdicts. Crucially, this was an audited coverage result rather than a failed request: all 100 rows record a clean `no Google fact-check match` reason, while 0/100 record an HTTP, API-key, or other request error. Agreement with the true labels is therefore not applicable (0/0), because there were no Google verdicts to compare (`google factchek/evaluation/google_escalation_results.csv`; `google factchek/evaluation/google_escalation_summary.txt`).

On this fixed bucket, the classifier alone correctly labelled 76 of 100 articles (76.0%), the previously committed DDG-only final labels correctly labelled 64 of 100 (64.0%), and the Google-first hybrid also correctly labelled 64 of 100 (64.0%). Since Google supplied no candidate on any row, every claim fell through to the live DDG + DeBERTa verifier. The unchanged aggregate accuracy therefore cannot be attributed to Google; it compares the stored outcome of the original DDG pass with the outcome of a later live fallback pass.

Those two DDG + DeBERTa passes produced markedly different verdict distributions. The original `evaluation/escalation_results.csv` contains 42 supported, 32 refuted, and 26 insufficient verdicts, whereas `google_escalation_results.csv` contains 30 supported, 12 refuted, and 58 insufficient verdicts. The shift from 42/32/26 to 30/12/58, despite identical final accuracy of 64.0% in both passes, demonstrates that the live-retrieval verdicts were unstable between runs and that aggregate accuracy alone obscures this behavior.

The `source=none` value on the 58 insufficient rows in the later CSV must not be interpreted as an absence of retrieved evidence. Each of those rows records `DDG evidence evaluated with DeBERTa NLI`, showing that evidence was retrieved and scored but did not clear the NLI verdict threshold; the verifier uses the distinct reason `no DDG evidence retrieved` for a true no-evidence outcome. Together, the zero Google coverage and unstable fallback distributions support keeping retrieved evidence as context rather than treating it as a reproducible automatic override. This conclusion is limited to the frozen bucket and the query-time responses committed in `034f8bf`, not to all Google Fact Check queries, datasets, or future runs.

## B.4 Limitations

Several limitations constrain the interpretation of these results. First, all classifier metrics were measured on the held-out WELFake split. They are not LIAR results and do not constitute a cross-dataset test. The near-ceiling classifier scores should therefore be interpreted as in-dataset performance rather than evidence of general fake-news detection under distribution shift (`docs/results-summary.md` §4).

Second, the escalation experiment covers only the 100 least-confident articles, not the full 9,398-article test set. It characterises the evidence layer where the classifier was most uncertain, but it does not estimate the effect of escalation across the entire held-out distribution. DDG retrieval is also live and can change between runs, so the evidence layer is not reproducible from the committed labels alone. The verdict-distribution shift documented in §B.3 makes this limitation observable rather than hypothetical (`docs/results-summary.md` §4; `google factchek/evaluation/google_escalation_results.csv`).

Third, MC Dropout improved accuracy while producing worse ECE than deterministic confidence in this run. MC averaging must therefore not be described as a calibration improvement. The discrepancy between discrimination and calibration requires further investigation before uncertainty estimates are used to govern higher-stakes routing decisions (`eval_MCFakeNews.ipynb`; `docs/writeup-facts-w.md` §1A).

Fourth, the explanation-faithfulness result is based on 8/10 faithful explanations from a sample of 10 assessed by a single reviewer. It provides a completed descriptive result, but no multi-reviewer agreement or larger-sample estimate is available. The row-8 verdict inversion also shows that a high aggregate faithfulness rate can coexist with a consequential evidence-conflict failure (`evaluation/faithfulness_review.csv`, commit `e529b46`).

Finally, the implemented NLI mapping is not safe as an automatic override. Taken together, the escalation breaks and explanation inversion indicate that evidence conflicts require explicit representation and validation. Treating a verdict as either a replacement label or post hoc support for a fixed label can erase the distinction between what the classifier predicted and what the external evidence actually indicates (`docs/results-summary.md` §3).

## B.5 Future work

Future work should first investigate why MC Dropout improved accuracy while its ECE was worse than deterministic confidence. The routing policy should not assume that improved classification accuracy implies improved calibration, and any uncertainty threshold should be validated against the decision it is intended to support (`eval_MCFakeNews.ipynb`; `docs/writeup-facts-w.md` §1A).

The explanation-faithfulness evaluation should be expanded beyond the initial 10-row sample. Multiple independent reviewers should apply a prespecified rubric, and the resulting analysis should report inter-rater agreement. The sample should deliberately include cases in which the NLI verdict conflicts with the classifier label, including the row-8 inversion pattern, because these cases directly test whether explanations preserve evidence direction instead of rationalising the classifier output (`evaluation/faithfulness_review.csv`, commit `e529b46`).

The escalation policy should likewise replace the unthresholded override with evidence-quality checks and validated score thresholds. Until such a policy demonstrates a net benefit, external evidence should support human review rather than automatically determine the final label. Evaluation should separately measure evidence relevance, verdict correctness, and the safety of applying a verdict when it conflicts with the classifier (`docs/results-summary.md` §3).

Google Fact Check coverage should be tested beyond the frozen bucket and with alternative claim-extraction and query formulations, because the committed experiment establishes 0/100 coverage only for those queries at that time. Cross-dataset evaluation on LIAR is also required to test whether the classifier and evidence-policy findings generalise beyond WELFake (`google factchek/evaluation/google_escalation_summary.txt`, commit `034f8bf`; `docs/results-summary.md` §4).

## B.6 Conclusion

The principal verified result is a RoBERTa-LoRA classifier with 0.9957 accuracy and 0.9957 macro F1 on 9,398 held-out WELFake articles, exceeding the TF-IDF baseline by 4.31 percentage points in accuracy. This establishes strong in-dataset performance while leaving cross-dataset generalisation unresolved (`docs/results-summary.md` §1, §4).

The evidence layer produced the report's central negative result. Automatic NLI override reduced low-confidence accuracy from 76.0% to 64.0%, so the deployed system appropriately keeps the classifier label final and presents NLI as context. The completed explanation review found 8/10 explanations faithful (80%), but the row-8 failure inverted a `supported` verdict to reinforce a fake label. Together with the `refuted` overrides that broke correct predictions, this failure shows the same evidence-conflict weakness in two independent components: both the decision policy and the explanation layer can misrepresent the role of evidence when it disagrees with the classifier (`docs/results-summary.md` §3, §4; `evaluation/faithfulness_review.csv`, commit `e529b46`).

The Google experiment added a second negative finding. Google Fact Check had zero coverage on the frozen bucket, while the repeated DDG fallback exposed live-retrieval verdict instability. These findings support retaining evidence as contextual information and motivate broader coverage and reproducibility studies (`google factchek/evaluation/google_escalation_results.csv`; `google factchek/evaluation/google_escalation_summary.txt`, commit `034f8bf`).

Despite these limitations, the system is operational as a public ZeroGPU Space. Its verified `/analyze` call completed in 64.4 seconds and returned REAL at 99.57% confidence, with Very stable prediction stability and an NLI verdict of Supported. The resulting contribution is therefore both a high-performing in-dataset classifier and an empirically grounded demonstration of why external evidence must be handled as a distinct, fallible signal rather than an automatic source of label authority (`docs/results-summary.md` §4).
