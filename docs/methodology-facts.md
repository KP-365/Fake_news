# Methodology Facts

This sheet is a source-backed reference for the report methodology section. Values were extracted from the executable code and preserved outputs in `scaffold_FakeNews_finn's_training.ipynb`, then checked against `predict.py` and `docs/results-summary.md`. It does not infer undocumented motivations.

## 1. Effective classifier configuration

| Choice | Implemented value | Source |
|---|---|---|
| Base model | `roberta-base` | Training notebook cells 17, 19, and 21; `predict.py::BASE_MODEL_NAME` |
| Task | Two-label sequence classification | Notebook cells 17 and 21; `predict.py::load_model()` |
| Number of labels | `2` | Notebook cells 17 and 21; `predict.py::load_model()` |
| Label IDs | `0 = real`, `1 = fake` | Notebook markdown cell 9 and model configuration in cells 17 and 21; `predict.py::ID_TO_LABEL` |
| PEFT method | LoRA | Notebook cell 17 |
| LoRA rank | `r=8` | Notebook cell 17 |
| LoRA alpha | `lora_alpha=16` | Notebook cell 17 |
| LoRA dropout | `lora_dropout=0.05` | Notebook cell 17 |
| LoRA target modules | `['query', 'value']` | Notebook cell 17 |
| LoRA bias handling | `bias='none'` | Notebook cell 17 |
| PEFT task type | `TaskType.SEQ_CLS` | Notebook cell 17 |
| Effective tokenizer | `roberta-base` tokenizer | Notebook cell 19; `predict.py` loads the saved checkpoint tokenizer |
| Maximum sequence length | `256` tokens | Notebook cell 19; `predict.py::MAX_LENGTH` |
| Token padding | `padding='max_length'` | Notebook cell 19; `predict.py::predict()` |
| Token truncation | Enabled | Notebook cell 19; `predict.py::predict()` |

**Why - RoBERTa with LoRA:** the notebook says the base weights should remain frozen and only low-rank LoRA updates should be trainable. Its preserved output reports 294,912 trainable parameters out of 124,942,082 total, or 0.2360%.

**Why - two labels and mapping:** the notebook explicitly samples both source labels for manual direction checking before training and configures the model with `id2label={0: 'real', 1: 'fake'}` and the inverse `label2id` mapping. `predict.py` repeats the same mapping.

**Why - rank, alpha, dropout, target modules, and bias:** the notebook records these settings but gives no separate rationale for the selected values.

**Why - `TaskType.SEQ_CLS`:** this matches the notebook's use of `AutoModelForSequenceClassification`; no additional rationale is stated.

**Why - maximum length 256:** the notebook records the limit but does not explain why 256 was selected. Both training and standalone prediction use the same limit.

### Tokenizer clarification

Notebook cell 17 briefly constructs `google-bert/bert-base-uncased` before training. Cell 19 then resets `model_name` to `roberta-base` and reassigns the tokenizer from that model before constructing all three datasets and data loaders. The effective training tokenizer is therefore the RoBERTa tokenizer, and `predict.py` loads the tokenizer saved in the trained checkpoint.

## 2. WELFake labels and source data

- The notebook loads WELFake from Kaggle dataset `saurabhshahane/fake-news-classification`.
- The raw WELFake frame preserved in notebook cell 11 has shape **72,134 rows by 4 columns**.
- Its columns are `Unnamed: 0`, `title`, `text`, and `label`.
- Before cleaning, the notebook reports **37,106 label-1 rows** and **35,028 label-0 rows**.
- The implemented semantic mapping is **0 = real** and **1 = fake**.
- The classifier is trained and evaluated on WELFake, not LIAR.

**Why - WELFake label inspection:** the notebook comments that the label direction must be confirmed before training and prints two examples from each class for manual inspection.

## 3. Cleaning and deduplication

The notebook applies these operations in order in cell 11:

1. Copy the loaded WELFake frame.
2. Drop `Unnamed: 0` when present because it is the stray source index column.
3. Print missing-value counts and the label distribution.
4. Print two title/text examples for each label to check label direction.
5. Fill missing titles with an empty string.
6. Drop rows whose `text` value is missing.
7. Apply the same `clean_text()` function to `title` and `text`:
   - return an empty string for non-string values;
   - remove URL-like text matching `http\S+|www\.\S+`;
   - remove HTML-like tags matching `<.*?>`;
   - collapse consecutive whitespace with `\s+` to one space;
   - strip leading and trailing whitespace.
8. Drop rows whose cleaned `text` has zero length.
9. Form `content` as `title + '. ' + text`, then strip leading and trailing periods and spaces with `.str.strip('. ')`.
10. Remove duplicate rows using `drop_duplicates(subset=['text'])`, so deduplication is based on article text rather than title or combined content.

The preserved notebook output reports:

- **558 missing titles** before filling;
- **39 missing text values** before dropping;
- **8,618 duplicate rows removed**;
- **62,649 rows** after cleaning and deduplication.

**Why - missing-value handling:** the notebook states that text matters more than title, so a missing title becomes an empty string while a row with no article text is removed.

**Why - retaining title:** the notebook states that headline sensationalism or clickbait can itself be a fake-news signal, so title and text are combined rather than discarding the title.

**Why - URL, HTML, whitespace, empty-text, and duplicate handling:** the code defines these operations but gives no additional prose rationale.

## 4. Deterministic train/validation/test split

The cleaned WELFake frame is split in two seeded, stratified steps:

```python
train_df, temp_df = train_test_split(
    df, test_size=0.3, stratify=df['label'], random_state=42
)
valid_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42
)
```

This yields:

| Partition | Share | Preserved row count |
|---|---:|---:|
| Training | 70% | 43,854 |
| Validation | 15% | 9,397 |
| Held-out test | 15% | 9,398 |
| Total | 100% | 62,649 |

The notebook writes the partitions as `welfake_train.csv`, `welfake_valid.csv`, and `welfake_test.csv` with `index=False`.

**Why - creating a split:** the notebook states that WELFake ships as one CSV, so explicit partitions must be created.

**Why - stratification:** both calls stratify on `label`, retaining the class distribution across partitions. The notebook does not separately explain the choice of the 70/15/15 proportions or seed 42.

## 5. Training configuration

| Choice | Implemented value | Source |
|---|---|---|
| Training batch size | `32` | Notebook cell 21 |
| Validation batch size | `32` | Notebook cell 21 |
| Test batch size | `32` | Notebook cell 21 |
| Training shuffle | `True` | Notebook cell 21 |
| Validation/test shuffle | `False` | Notebook cell 21 |
| Optimizer | `torch.optim.AdamW` | Notebook cell 21 |
| Learning rate | `2e-4` | Notebook cell 21 |
| Weight decay | `0.01` | Notebook cell 21 |
| Epoch cap | `3` | `num_epochs=3` in notebook cell 21 |
| Early-stopping metric | Full validation-set mean loss at each epoch end | Notebook cell 21 |
| Early-stopping patience | `1` non-improving epoch | Notebook cell 21 |
| Best-state representation | Deep copy of `get_peft_model_state_dict(model)` | Notebook cell 21 |
| Best-state restoration | `set_peft_model_state_dict(model, best_model_state)` before test evaluation and save | Notebook cell 21 |
| Progress-loss interval | Every `50` training steps | Notebook cell 21 |
| Interim validation interval | Every `300` training steps after step 0 | Notebook cell 21 |
| Interim validation cap | First `50` validation batches | Notebook cell 21 |

The interim 50-batch validation checks are progress heartbeats only. Early stopping compares the full validation mean loss computed at the end of each epoch.

The preserved run completed all three allowed epochs. Full validation loss changed from **0.0290** after epoch 1 to **0.0119** after epoch 2 and **0.0139** after epoch 3. Epoch 3 was the first non-improving epoch, so patience 1 triggered early stopping and the best saved adapter state was restored before test evaluation.

**Why - batch size 32, learning rate `2e-4`, AdamW, weight decay `0.01`, and three-epoch cap:** these values are explicit in the code, but the notebook does not provide separate rationales for choosing them.

**Why - training shuffle only:** the notebook comments that training samples are shuffled each epoch; validation and test loaders remain deterministic. No further rationale is stated.

**Why - validation-loss early stopping and restore:** the code selects the adapter state with the lowest full validation loss and restores it before testing and saving, preventing the final non-improving epoch from becoming the deployed state. The notebook does not add a separate prose explanation.

## 6. Standalone inference parity in `predict.py`

`predict.py` confirms the deployed inference path uses:

- `BASE_MODEL_NAME = 'roberta-base'`;
- `MAX_LENGTH = 256`;
- `ID_TO_LABEL = {0: 'real', 1: 'fake'}`;
- two sequence-classification labels and the inverse `label2id` mapping;
- the tokenizer saved under `models/roberta-trained-welfake`;
- the PEFT adapter loaded onto CPU first;
- the separately saved `classifier_head.pt`, loaded with `map_location='cpu'` and `weights_only=True`;
- max-length padding and truncation during tokenization;
- softmax over the two logits, with `argmax` selecting the output label and the selected probability returned as confidence.

On normal local hardware, inference selects CUDA when available and otherwise CPU. On a ZeroGPU Space startup, `SPACES_ZERO_GPU` forces initial CPU loading so CUDA work can be moved into allocated GPU functions later.

## 7. Cross-check against `docs/results-summary.md`

The methodology values agree with the committed results summary:

- Both sources specify the deterministic, stratified **70/15/15** WELFake split with **seed 42**.
- Both report a held-out test size of **9,398** articles.
- The results summary gives test support as **5,193 real** and **4,205 fake**, totaling 9,398 and using the same `0 = real`, `1 = fake` mapping.
- The training notebook's preserved test accuracy is **0.9957**, matching the results summary.
- The notebook's `f1_score` call uses scikit-learn's default binary averaging, so its preserved **0.9952** value is the fake-class F1, not macro-F1. The results summary separately reports **0.9952 fake-class F1** and **0.9957 macro-F1**. These values are consistent rather than contradictory.
- The results summary's confusion matrix is `[[5178, 15], [25, 4180]]`; its row totals are 5,193 real and 4,205 fake, consistent with the stated class supports and label order.

## 8. Reporting boundaries

- Describe the implemented model as **RoBERTa-base with a LoRA sequence-classification adapter trained on WELFake**.
- Do not describe it as BERT trained on LIAR; LIAR is downloaded and inspected but is not used by the implemented training path.
- Describe `0.9952` as the notebook's binary fake-class F1, not macro-F1.
- Use `0.9957` for the separately evaluated macro-F1 reported in `docs/results-summary.md`.
- Do not invent motivations for hyperparameters where the notebook provides only the value.
