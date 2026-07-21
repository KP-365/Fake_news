# Google Fact Check extension

This project keeps the original uncertainty-aware pipeline and adds Google Fact Check as the first verification route.

## Final flow

1. RoBERTa-LoRA classifies the WELFake test split.
2. MC Dropout runs 30 stochastic passes.
3. The 100 lowest-confidence articles are escalated.
4. Google Fact Check Tools API is searched first.
5. A sufficiently similar, unambiguous professional rating returns `supported` or `refuted`.
6. Missing, weak, ambiguous, or conflicting Google results fall back to DDG retrieval and DeBERTa NLI.
7. `insufficient` keeps the original RoBERTa label.

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_FACTCHECK_API_KEY="YOUR_KEY"
python evaluation/eval_escalation.py
```

For Colab:

```python
import os
from google.colab import userdata
os.environ["GOOGLE_FACTCHECK_API_KEY"] = userdata.get("GOOGLE_FACTCHECK_API_KEY")
```

Store the key in Colab Secrets under `GOOGLE_FACTCHECK_API_KEY`.

## Output

`evaluation/escalation_results.csv` includes:

- classifier label and confidence
- verification source (`google_fact_check`, `ddg_deberta`, or `none`)
- Google match score, rating, publisher, matched claim, and review URL
- DeBERTa entailment and contradiction scores where DDG fallback is used
- final label and whether escalation changed the classifier decision
- full verification payload as JSON for auditability

## Conservative Google matching

Google results are only used when:

- the claim similarity score is at least `0.45`; and
- the publisher's textual rating maps clearly to supported or refuted; and
- the strongest similar reviews do not conflict.

Tune `GOOGLE_MIN_MATCH_SCORE` in `verify.py` during validation rather than on the held-out test labels.
