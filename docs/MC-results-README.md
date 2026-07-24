# Monte Carlo Dropout — Evaluation Results

These files are produced by [`eval_MCFakeNews.ipynb`](../eval_MCFakeNews.ipynb) (open a GPU
runtime, then **Runtime → Run all**). They record the Monte Carlo (MC) Dropout uncertainty
evaluation of the fine-tuned RoBERTa-LoRA classifier on the held-out **9,398-article** WELFake
test split, using **30 stochastic forward passes** per article with dropout kept active.

The MC prediction is the argmax of the mean softmax over the 30 passes; confidence is the
maximum mean class probability. Entropy quantities are in **nats** (natural log), so the maximum
possible entropy for two classes is `ln 2 ≈ 0.693`.

## File index

| File | What it is |
|---|---|
| `confusion_matrix.png` | Confusion matrix of the MC-averaged predictions (true labels as rows) |
| `classifier_metrics.json` | Accuracy, macro-F1, per-class precision/recall; MC vs deterministic; mean predictive entropy and mutual information |
| `uncertainty_rejection.png` | Predictive-entropy boxplot (correct vs incorrect) + rejection / accuracy–coverage curve |
| `uncertainty_summary.json` | Mean predictive entropy and mutual information, split by correct vs incorrect predictions |
| `rejection_curve.npz` | Raw arrays: `coverage`, `retained_accuracy` (to re-plot the rejection curve) |
| `reliability_diagram.png` | Reliability diagram, MC vs deterministic |
| `calibration.json` | Expected Calibration Error (ECE) for MC and deterministic predictions |
| `sample_predictions.csv` | 10 seeded sample articles with true/predicted label, confidence, entropy, mutual information |
| `per_article_results.csv` | Full per-article results: title, true/predicted label, confidence, predictive entropy, mutual information |
| `mc_arrays.npz` | Raw NumPy arrays for every quantity (`mean_probabilities`, `predictive_entropy`, `expected_entropy`, `mutual_information`, `predicted_labels`, `true_labels`) — for full reproducibility |

## What each metric tells you

- **Predictive entropy** — total uncertainty of the mean prediction (higher = less certain).
- **Expected entropy** — the aleatoric part (average per-pass entropy).
- **Mutual information** — the epistemic part (`predictive − expected`): disagreement *between*
  the 30 passes.
- **Rejection curve** — accuracy on the retained set as the most-uncertain articles are deferred;
  a curve that climbs as coverage drops means uncertainty reliably flags the errors.
- **ECE** — average gap between confidence and accuracy across bins (lower = better calibrated).

## Regenerating these files

Open `eval_MCFakeNews.ipynb` on a GPU runtime and run all cells. The notebook writes every file
above into this folder and (optionally) commits them to the repository. MC Dropout is stochastic,
so small run-to-run variation in the entropy / MI values is expected; the qualitative results
(accuracy, calibration, and rejection behaviour) are stable.
