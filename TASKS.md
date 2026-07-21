# Task Tracker

This checklist reflects evidence currently committed in the repository. Checked items are
implemented; unchecked items remain incomplete. Proposal targets that were replaced by a
different implementation are labelled as deviations rather than silently treated as complete.

Legend: **K** = Kayleb Parkes, **E** = Eric Kamalendran, **W** = William McKie

## Milestone 0: Repository and data

- [x] Repository scaffold, dependency environment, and Colab setup - K
- [x] Download and inspect LIAR and WELFake - E
- [x] Commit the deployable RoBERTa adapter, classifier head, and tokenizer - K

## Milestone 1: Classification and uncertainty

- [x] Fine-tune `roberta-base` with LoRA on WELFake - K
  - *Proposal deviation: the proposal specified BERT on LIAR.*
- [x] Recreate a deterministic stratified 70/15/15 WELFake split - K
- [x] Add validation-loss early stopping and best-adapter restoration - K
- [x] Provide standalone classifier inference in `predict.py` - K
- [x] Report accuracy, per-class precision/recall, macro-F1, and confusion matrix on WELFake - K
- [x] Compare against TF-IDF logistic regression and a majority-class baseline - W
- [x] Implement 30-pass MC Dropout uncertainty estimation - K
- [x] Produce reliability/ECE and accuracy-versus-coverage diagnostics - K
- [ ] Train and evaluate the proposed LIAR classifier
- [ ] Complete classifier Methodology, Implementation, and Results report sections

## Milestone 2: Evidence verification and escalation

- [x] Implement WELFake cleaning, deduplication, and RoBERTa tokenisation - E
- [ ] Build one shared LIAR/WELFake preprocessing pipeline
- [x] Implement DDG evidence retrieval with bounded timeouts in `verify.py` - E
  - *Proposal deviation: DDG remains the deployed retrieval source; Google Fact Check was later measured in the fixed-bucket experiment (`034f8bf`).*
- [x] Implement zero-shot DeBERTa NLI verdicts: supported/refuted/insufficient - E
- [x] Gate the 100 lowest-confidence MC Dropout cases into verification - E
- [x] Map NLI verdicts to a final label and write per-article escalation CSV rows - E
- [x] Complete and preserve a full 100-article escalation run
- [x] Analyse verdict distribution and classifier/NLI disagreements
- [x] Measure evidence-retrieval coverage with the fixed-bucket Google Fact Check experiment - evidence commit `034f8bf`
- [x] Complete verification Results/Discussion outline and prose, including Conclusions - evidence commit `9cf1600`
- [ ] Complete verification Related Work section

## Milestone 3: Explanation, baselines, and interface

- [x] Add baseline model comparison and evaluation chart - W
- [x] Implement numbers-only Anthropic explanations in `explain.py` - W
- [x] Prevent the explanation prompt from reclassifying or second-guessing the fixed label - W
- [x] Connect `explain.py` to a single end-to-end classifier → verifier → explainer command - W
- [x] Build a local Gradio interface - W
- [x] Deploy publicly on Hugging Face Spaces - W
- [ ] Produce presentation slides
- [ ] Complete Introduction, Abstract, Evaluation, Contributions, and formatting

## Milestone 4: Integration and submission

- [x] Review a sample of generated explanations for faithfulness to the structured signals - **8/10 faithful**; evidence commit `e529b46`
- [x] Run and document an end-to-end demo test
- [x] Clearly scope conclusions to WELFake; cross-dataset evaluation remains future work - evidence commit `9cf1600`
- [ ] Assemble the final report and submission checklist

## Current scope

The validated implementation is a **RoBERTa-LoRA classifier trained and tested on WELFake**,
with MC Dropout evaluation, DDG + DeBERTa verification, a numbers-only Anthropic explanation
function, a local Gradio interface, and a public ZeroGPU Space. It is not the proposed
BERT-on-LIAR system.
