# Report Outline: Results and Discussion

**Owner:** William  
**Purpose:** Full outline for the Results and Discussion sections. Every number below is pulled verbatim from `docs/writeup-facts-w.md` and `docs/results-summary.md`, with the backing artifact cited next to it. Nothing here is invented; pending work is marked as a TODO block, not written up.

**How to use:** this is an outline, not prose. Each bullet states the point to make and the artifact that backs it. Belongs-elsewhere notes flag content that lives in the Methodology section rather than Results or Discussion.

---

## Part A: Results

### A.1 Experimental setup

- State the evaluation basis: deterministic, stratified **70/15/15** WELFake split with **seed 42**; held-out test set of **9,398** articles (**5,193 real**, **4,205 fake**), none used for training (`docs/results-summary.md` §1; `docs/writeup-facts-w.md` §1).
- State that classifier metrics are in-dataset on this held-out WELFake split, not LIAR or a cross-dataset test (`docs/results-summary.md` §4).
- **Belongs elsewhere (Methodology):** the model configuration (RoBERTa-base + LoRA), training hyperparameters, cleaning/dedup steps, and split construction belong in Methodology. Cite `docs/methodology-facts.md`; do not restate them here beyond one framing sentence.

### A.2 Classifier performance

- Report RoBERTa-LoRA held-out metrics: Accuracy **0.9957**, Macro precision **0.9958**, Macro recall **0.9956**, Macro F1 **0.9957** (`docs/results-summary.md` §1).
- Report per-class results: Real - precision **0.9952**, recall **0.9971**, F1 **0.9962**, support **5,193**; Fake - precision **0.9964**, recall **0.9941**, F1 **0.9952**, support **4,205** (`docs/results-summary.md` §1).
- Present the confusion matrix (true labels as rows, real/fake order): `[[5178, 15], [25, 4180]]` (`docs/results-summary.md` §1; `docs/writeup-facts-w.md` §1).
- Note the error profile: **40** errors total - **15** real articles labelled fake, **25** fake articles labelled real (`docs/results-summary.md` §1).
- Figure: confusion-matrix plot preserved in `eval_FakeNews.ipynb` (cited via `docs/results-summary.md`).

### A.3 Baseline comparison

- Report the comparison table on the same held-out split: RoBERTa-LoRA **0.9957** accuracy / **0.9957** macro F1; TF-IDF + logistic regression **0.9526** / **0.9521**; majority-class baseline **0.5526** / **0.3559** (`docs/results-summary.md` §1; `docs/writeup-facts-w.md` §1).
- State the margins: RoBERTa-LoRA exceeds TF-IDF by **4.31** percentage points accuracy and **4.36** points macro F1 (`docs/results-summary.md` §1; `docs/writeup-facts-w.md` §1).
- Figure: macro-F1 comparison bar chart preserved in `eval_FakeNews.ipynb` (cited via `docs/writeup-facts-w.md` §1).
- **Belongs elsewhere (Methodology):** baseline implementation details - TF-IDF `max_features=30000`, `ngram_range=(1, 2)`, `min_df=2`; logistic regression `max_iter=1000`, `random_state=42`, `solver="liblinear"`; `DummyClassifier(strategy="most_frequent")` - belong in Methodology, cited to `eval_FakeNews.ipynb` via `docs/writeup-facts-w.md` §1.
- Caveat for honesty: no dedicated baseline CSV is committed under `evaluation/`; the only committed evaluation CSV is `escalation_results.csv` (`docs/writeup-facts-w.md` §5). The table values come from the notebook's preserved output.

### A.4 MC Dropout calibration

> **TODO - pending Kayleb's notebook run.**
> `eval_MCFakeNews.ipynb` contains the implemented 30-pass MC Dropout evaluation, predictive-entropy and mutual-information diagnostics, a rejection curve, a 15-bin reliability diagram, and ECE calculations for MC-averaged and deterministic confidence (`docs/results-summary.md` §2).
> The committed notebook does **not** contain executed outputs for the MC inference, calibration, or ECE cells. There is no numerical MC Dropout ECE, deterministic ECE, entropy, or accuracy-versus-coverage result available to quote (`docs/results-summary.md` §2; `docs/writeup-facts-w.md` §5).
> **Do not write any calibration number until the notebook is fully executed and its outputs are committed.** When that lands, this subsection reports: MC vs deterministic accuracy/macro-F1, mean predictive entropy, mean mutual information, reliability diagram, ECE values, and the rejection (accuracy-versus-coverage) curve.

### A.5 Escalation experiment

- Describe the design: `evaluation/escalation_results.csv` holds the **100** WELFake test articles selected as least confident by MC Dropout; DDG retrieval plus DeBERTa NLI produced **42 supported**, **32 refuted**, **26 insufficient** verdicts (`docs/results-summary.md` §3).
- State the verdict rule: `supported` maps to real, `refuted` maps to fake, `insufficient` keeps the classifier label (`docs/results-summary.md` §3).
- Report the headline result: classifier-only **76.0% (76/100)** vs escalated **64.0% (64/100)** - a **12**-percentage-point loss (`docs/results-summary.md` §3).
- Report the per-verdict table:

| Verdict | Rows | Classifier accuracy | Actual overrides | Fixed | Broken | Final accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Supported | 42 | 73.8% (31/42) | 14 | 7 | 7 | 73.8% (31/42) |
| Refuted | 32 | 71.9% (23/32) | 20 | 4 | 16 | 34.4% (11/32) |
| Insufficient | 26 | 84.6% (22/26) | 0 | 0 | 0 | 84.6% (22/26) |
| **Overall** | **100** | **76.0% (76/100)** | **34** | **11** | **23** | **64.0% (64/100)** |

  (`docs/results-summary.md` §3; counts independently recomputed from `evaluation/escalation_results.csv` per `docs/writeup-facts-w.md` §2.)

- Note CSV-derived label counts (not in the summary but directly derivable): 66 true-real / 34 true-fake rows; 62 classifier-real / 38 classifier-fake; 56 final-real / 44 final-fake (`docs/writeup-facts-w.md` §2).
- Scope caveat: this covers the 100 least-confident test articles, not the full 9,398 (`docs/results-summary.md` §4).

### A.6 Deployed Space verification

- Present the live ZeroGPU Space at `https://wf1212-fake-news-detector.hf.space` (`docs/results-summary.md` §4).
- Report the verified `gradio_client` call to the named `/analyze` endpoint on 20 July 2026: **64.4 seconds** wall-clock; final classifier label **REAL** at **99.57%** confidence; prediction stability **Very stable**; NLI verdict **Supported**; runtime includes live DDG retrieval and two ZeroGPU allocations (`docs/results-summary.md` §4; `docs/writeup-facts-w.md` §4).
- Note the caveat: no permanent transcript of the full response payload is committed; report the summary values only (`docs/writeup-facts-w.md` §5). The repository does not commit a runtime-status artifact (`docs/writeup-facts-w.md` §5).
- **Belongs elsewhere (Methodology):** Space configuration (`sdk: gradio`, `sdk_version: 6.20.0`, `python_version: 3.12.12`, `app_file: app.py`), the 5,000-character input bound, and per-request Anthropic-key handling are implementation details belonging to Methodology/Implementation, cited to `space/README.md` and `app.py` via `docs/writeup-facts-w.md` §4.

### A.7 Explanation faithfulness

> **TODO - pending the 10-row human review.**
> `eval_faithfulness.ipynb` provides the manual review workflow: sample 10 escalation rows with `np.random.default_rng(42)`, regenerate explanations from structured signals, and export `evaluation/faithfulness_review.csv` (`docs/writeup-facts-w.md` §3).
> No completed `evaluation/faithfulness_review.csv` is committed, and no explanation-faithfulness or human-evaluation result has been completed (`docs/writeup-facts-w.md` §5; `docs/results-summary.md` §4).
> **Do not report a faithfulness rate until the review CSV is committed.** When it lands, this subsection reports the yes/no match rate across the 10 rows and discusses failure cases.

---

## Part B: Discussion

### B.1 Interpreting near-ceiling in-dataset accuracy

- The central tension: **0.9957** accuracy / **0.9957** macro F1 on held-out WELFake is near-ceiling (`docs/results-summary.md` §1).
- Point to the error profile as the honest detail: only **40** errors, split **15** false-fake / **25** false-real (`docs/results-summary.md` §1).
- Discuss against baselines: the TF-IDF margin (**4.31** points accuracy) shows the transformer adds value beyond word-frequency patterns (`docs/results-summary.md` §1).
- Frame the limit: these are in-dataset results on one corpus; the reporting boundaries state they are not LIAR or cross-dataset (`docs/results-summary.md` §4). Argue that WELFake's cleaned, label-verified construction likely contributes to the ceiling, but present this as interpretation, not a measured claim - no cross-dataset artifact exists to test it (`docs/results-summary.md` §4; `docs/writeup-facts-w.md` §5).

### B.2 Why NLI override failed

- The evidence: escalation cut low-confidence accuracy from **76.0%** to **64.0%** across **34** label changes - **11** fixes vs **23** breaks (`docs/results-summary.md` §3).
- Identify the failure mode: `refuted` verdicts, **4** fixes vs **16** breaks, dropping that group from **71.9%** to **34.4%**; `supported` was neutral (**7**/**7**); `insufficient` correctly kept labels and had the best classifier accuracy at **84.6%** (`docs/results-summary.md` §3).
- Explain the mechanism (as documented interpretation): web snippets can be irrelevant, ambiguous, or about a related claim, and contradiction scores do not by themselves establish article-level falsity; the unthresholded verdict-to-label mapping is unsafe as an automatic override (`docs/results-summary.md` §3).
- Avoid overclaiming: the results do not show retrieval/NLI is useless - they show the current mapping is unsafe; better suited to human review or a future escalation policy with evidence-quality checks and validated thresholds (`docs/results-summary.md` §3).
- Live-data caveat: DDG results are live web data and can change between runs (`docs/results-summary.md` §4).

### B.3 Google Fact Check API omission

> **TODO - pending Eric's coverage test.**
> The `google factchek/` directory (Eric's commits) and its `README_GOOGLE_FACTCHECK.md` exist in the repository, but no coverage or comparison result is committed in the evaluation artifacts (`docs/writeup-facts-w.md` artifact map; repository `google factchek/` directory).
> **Do not write claims about Google Fact Check coverage, agreement rates, or why it was omitted until Eric's coverage test is committed.** When it lands, this subsection reports what the API covers relative to WELFake claims and justifies the DDG retrieval choice with evidence rather than assertion.

### B.4 Limitations

- **In-dataset only:** all classifier metrics are WELFake held-out, not LIAR or cross-dataset (`docs/results-summary.md` §4).
- **Escalation scope:** the experiment covers the 100 least-confident articles, not the full 9,398 (`docs/results-summary.md` §4).
- **Live retrieval:** DDG results change between runs; the evidence layer is not reproducible as committed (`docs/results-summary.md` §4).
- **Missing calibration numbers:** no numerical ECE, entropy, or coverage result is preserved (`docs/results-summary.md` §2, §4).
- **Missing faithfulness evaluation:** no human-explanation result completed (`docs/results-summary.md` §4).
- **Unsafe override mapping:** the verified NLI layer cannot safely override the classifier as implemented (`docs/results-summary.md` §3).

### B.5 Future work

- Complete and commit the MC Dropout calibration run (ECE, entropy, coverage) so calibration can be reported (TODO from `docs/results-summary.md` §2).
- Complete the 10-row explanation-faithfulness review and commit `evaluation/faithfulness_review.csv` (TODO from `docs/writeup-facts-w.md` §3, §5).
- Design a safer escalation policy: evidence-quality checks and validated score thresholds rather than an unthresholded override; position the current output for human review (`docs/results-summary.md` §3).
- Complete Eric's Google Fact Check coverage test to ground the retrieval-source choice (TODO from §B.3 above).
- Evaluate cross-dataset generalization (LIAR), which the current artifacts do not cover (`docs/results-summary.md` §4).

### B.6 Conclusion

- Lead with the verified result: a RoBERTa-LoRA classifier achieving **0.9957** accuracy and macro F1 on 9,398 held-out WELFake articles, **4.31** points above a TF-IDF baseline (`docs/results-summary.md` §1).
- Be honest about the evidence layer: automatic NLI override made low-confidence articles worse (**76.0% → 64.0%**), so the deployed system keeps the classifier label final and treats NLI as context only (`docs/results-summary.md` §3, §4).
- Close on the working system: a publicly deployed ZeroGPU Space verified end-to-end via its `/analyze` endpoint (**64.4** seconds, REAL at 99.57%, Very stable, NLI Supported) (`docs/results-summary.md` §4).
- Acknowledge the two open evaluation items (calibration numbers, faithfulness review) as committed next steps rather than completed contributions (`docs/results-summary.md` §2, §4).

---

## Belongs-elsewhere summary (Methodology material, not Results/Discussion)

- Model/LoRA configuration, tokenizer, max sequence length, training hyperparameters, early stopping → Methodology (`docs/methodology-facts.md`).
- WELFake cleaning/dedup steps and split construction → Methodology (`docs/methodology-facts.md`).
- TF-IDF/logistic/Dummy baseline hyperparameters → Methodology (`eval_FakeNews.ipynb` via `docs/writeup-facts-w.md` §1).
- Space SDK/Python versions, input bound, key handling → Implementation/Methodology (`space/README.md`, `app.py` via `docs/writeup-facts-w.md` §4).
- Explanation prompt internals (seven structured signals, `claude-haiku-4-5`, `max_tokens=220`, `temperature=0`) → Methodology/Implementation (`explain.py` via `docs/writeup-facts-w.md` §3); only the faithfulness *result* belongs in Results once the review exists.
