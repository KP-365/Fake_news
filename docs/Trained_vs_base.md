# Trained vs Base — Fine-Tuning Ablation

Explanation of [`Trained_vs_base.ipynb`](https://github.com/KP-365/Fake_news/blob/main/Trained_vs_base.ipynb).

## What this notebook answers

**Does the LoRA fine-tuning actually create the capability, or is `roberta-base`'s pretraining
already doing the work?** This notebook isolates the effect of fine-tuning by comparing the
fine-tuned model against an otherwise-identical model that was **not** fine-tuned.

It does **not** train anything — it loads the already-trained checkpoint and evaluates it
side-by-side with a fresh base model.

## Why this is an *ablation*, not a baseline

- The **base model** is `roberta-base` with a **randomly initialised** classification head — the
  encoder's pretraining is intact, but nothing has been trained on WELFake, so its head predicts
  essentially at chance.
- It uses the **same architecture and the same tokenizer** as the trained model, so the **only**
  difference between the two is the LoRA fine-tuning. That controlled setup is what makes it an
  *ablation*: remove fine-tuning, hold everything else fixed, measure the drop.
- This is different from a **competitive baseline** (TF-IDF + Logistic Regression, majority-class),
  which lives in `eval_FakeNews.ipynb §5`. Those answer "is the transformer worth it vs. simple
  methods?"; this answers "is the *fine-tuning* worth it vs. the raw pretrained model?".

## Method

| Component | Trained model | Base model |
|---|---|---|
| Encoder | `roberta-base` | `roberta-base` |
| Classification head | fine-tuned on WELFake (LoRA) | **randomly initialised** (not trained) |
| Tokenizer | RoBERTa tokenizer | RoBERTa tokenizer |
| Fine-tuned on WELFake? | ✅ | ❌ |

Both models are evaluated on the **same held-out WELFake test split** (9,398 articles, stratified
70/15/15, seed 42) with an identical inference pipeline: `max_length=256`, batch size 32, softmax
over the two classes, argmax as the prediction and the max probability as confidence.

## Cell-by-cell

| Cell | What it does |
|---|---|
| **1. Setup** | `git clone` + `pip install` + torchvision fix |
| **2. Load trained model** | `load_model()` → the fine-tuned model (`model`, `tokenizer`, `device`) |
| **3. Test split** | Reproduce the WELFake cleaning + stratified 70/15/15 split (seed 42) → `test_df` |
| **4. Load base model** | Fresh `roberta-base` with a random classification head + RoBERTa tokenizer (`o_model`, `o_tokenizer`) — the "before fine-tuning" reference |
| **5. Evaluate trained** | `evaluate_model` / `score` helpers → predictions + `classification_report` |
| **6. Evaluate base** | Same pipeline on the base model → predictions + `classification_report` |
| **7. Compare** | Side-by-side metrics table (accuracy, precision, recall, F1, and the accuracy gain over base) + paired confusion matrices with accuracy in each title |

## How to read the output

- The **comparison table** lists both models' accuracy / precision / recall / F1 plus an
  `accuracy_gain` column (trained minus base).
- The **paired confusion matrices** show the error structure of each model, with accuracy in the
  subplot titles.
- **Expected pattern:** the fine-tuned model scores ~0.99 (matching the `eval_FakeNews.ipynb`
  headline of 0.9957), while the base model sits near chance / heavily skewed, because its head is
  untrained.

## What it demonstrates

The fine-tuned model vastly outperforms the raw base model, which shows that the **LoRA fine-tuning
is what creates the fake-news classification capability** — not `roberta-base`'s language
pretraining alone. Because LoRA trains only a tiny adapter (~294,912 parameters on top of the frozen
~109.8M encoder), this also demonstrates that a very small, cheap adapter is sufficient to unlock
strong task performance.

## Using it in the report

- Place this in **Results & Discussion** as an *ablation study* supporting the Methodology's claim
  that LoRA fine-tuning is the mechanism doing the work.
- **Be honest about the framing:** the base model's head is untrained, so a near-chance score is
  *expected* — this is a controlled sanity/ablation result, not a strong external comparison. The
  "is the model worth it vs. simple methods?" argument comes from the TF-IDF / majority-class
  baselines in `eval_FakeNews.ipynb`, which you should cite alongside this.

## How to run

Open the notebook on a Colab **GPU** runtime and **Runtime → Run all**. It downloads
`roberta-base` for the base model and loads the LoRA checkpoint from `models/roberta-trained-welfake/`
for the trained model.
