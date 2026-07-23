# Report integration punch-list

Use this list when merging William's Results and Discussion and Conclusions into the existing `v2-gc.pdf` sections.

1. **Experiments.** Add one or two sentences introducing the Google-first experiment as a proposal-fidelity check before accepting the DDG pivot.
2. **Proposed Method §3.5 and Results and Discussion.** Reconcile Method's confidence-threshold routing with Results' fixed 100-article bucket. State that the threshold was deliberately left unvalidated, so the evaluation used a reproducible fixed bucket instead.
3. **Proposed Method §3.7 Baseline.** Fill the empty subsection because Table 1 reports TF-IDF and majority-class baseline rows.
4. **Proposed Method §3.5.** Repair the truncated sentence ending with “A fixed offline”.
5. **References.** Remove the duplicate Gal and Ghahramani reference.
6. **Front matter, Introduction, and Related Work.** Delete the title and author placeholders, sample subsection text, and Related Work example figures and table.
7. **Experiments.** Restate the proposal's five success criteria verbatim because `project_proposal.docx` was deleted and the Results verdict paragraph judges against them.
8. **Abstract and Related Work.** Describe the deployed no-gate design: verification runs on every input as context, while selective escalation is an offline experiment.
9. **Abstract.** Remove or correct the claim of evaluation on LIAR; LIAR was inspected but never used for training or evaluation.
10. **References.** Unify citations in one style backed by `docs/candidate-citations.bib`, and verify every `\cite` key in `report-results-draft-v3.md` resolves.
