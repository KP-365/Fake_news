# A Three-Layer Framework for Fake News Detection, Verification, and Explanation

**Authors**

- First Author — first.author@qmul.ac.uk
- Second Author — second.author@qmul.ac.uk
- Eric Kamalendran — ec251192@qmul.ac.uk

## Abstract
The proliferation of misinformation on online platforms has led to the need for automated systems to effectively
identify fake news and generate trustworthy predictions. Traditional deep learning models are prone to overconfident
predictions, even on ambiguous and unseen data. This project investigates the potential to improve the robustness of
automated fake news detection by integrating uncertainty estimation and evidence-based verification. We propose a
hybrid framework that combines a fine-tuned RoBERTa language model with Monte Carlo (MC) Dropout to
estimate prediction uncertainty.

Articles identified as low-confidence are escalated to a retrieval-based verification stage on a selective basis.
Relevant evidence is retrieved from online sources and assessed by a DeBERTa natural language inference model to
decide if the evidence supports, refutes, or is insufficient to verify the claim. The framework is evaluated on the
WELFake and LIAR dataset with standard classification metrics and analysis of uncertainty-guided verification.
This work combines transformer-based classification, uncertainty estimation, and retrieval-augmented evidence
verification to improve the robustness, interpretability, and reliability of fake news detection systems, with the aim
of reducing the risk of confidently misclassifying uncertain information. Our proposed framework offers a practical
path to more trustworthy AI systems for misinformation detection.

## Introduction
The rapid growth of digital media and social networking platforms has changed the way news and information are
shared. While making information more accessible, it has also increased the spread of misinformation and fake
news, bringing with it considerable social, political and economic challenges. Therefore, automatic discrimination of
real news from fake news has become an important research area in Artificial Intelligence (AI) and Natural
Language Processing (NLP).
For example, recent developments in transformer based language models, like RoBERTa, have significantly
improved the performance of automated fake news classification. However, these models often make over-confident
predictions even when faced with ambiguous or unseen information. In reality, relying on a single classification
could lead to less reliable and trustworthy AI systems, particularly when incorrect predictions are made with high
confidence.

The project proposes a three-stage framework to improve the reliability of automated detection of fake news. First, a
fine-tuned RoBERTa model predicts whether a news article is real or fake and calculates the confidence of the
model. Simultaneously there is a retrieval of relevant web evidence using the DeBERTa Natural Language Inference
(NLI) model to determine whether the evidence supports, refutes or is insufficient to verify the original claim. The
data feeds into a LLM in order to output an explanation of the decision, providing more transparency.

The main goal of this work is to explore whether the combination of transformer-based classification,
uncertainty-guided evidence verification and explainable decision-making can enhance the reliability of automated
fake news detection. The proposed framework is evaluated with the WELFake dataset and evaluated with standard
classification metrics such as accuracy, precision and recall The findings contribute to building more trustworthy and
interpretable AI systems for misinformation detection and highlight the value of incorporating uncertainty-aware
verification in modern fake news classification pipelines.

## Related Work
With the surge of social media and online news, the pace of real news has accelerated, yet the distribution of fake
news has also reached larger masses. The damages that fake news has brought to politics, healthcare, and public
safety have initiated the need for the creation of reliable automated systems to address this problem. Research from
Vosoughi, Roy and Aral (2018) has shown that, compared to real news, the spread of social media fake news is a
larger and much more widespread phenomenon. This has also helped detect techniques for misleading news. For this
reason, the fraudulent news detection has been an emerging and important field of study in Natural Language
Processing (NLP). This trend has seen a clear transition from machine learning as the primary technique used, to the
development of deep learning frameworks and Transformer-based architectures that better capture and understand
contextual language.

Early fake news detection systems were primarily based on traditional machine learning algorithms, including
Support Vector Machines (SVM), Naïve Bayes, Logistic Regression and Random Forests. These approaches
extracted handcrafted linguistic, syntactic and semantic features followed by classification (Shu et al., 2017).
Though these models showed the capability of classifying fake and genuine news based on the text properties, they
were heavily dependent on the manual feature engineering. Moreover, handcrafted features often failed to generalise
to other domains and writing styles, making traditional approaches less effective in the face of evolving
misinformation. Such limitations led to the shift towards deep learning models that can automatically learn
contextual representations from text.

The advent of the Transformer architecture revolutionised natural language processing by replacing recurrent neural
networks with self-attention mechanisms that efficiently capture long-range contextual dependencies (Vaswani et al.,
2017). Based on this architecture, BERT proposed bidirectional contextual representation which brought a huge
performance gain on many language understanding tasks (Devlin et al., 2019). RoBERTa improved upon BERT by
removing the next sentence prediction objective, increasing training data, and optimising the pre-training procedure,
resulting in consistently higher classification accuracy across a variety of NLP benchmarks (Liu et al., 2019).
Despite these improvements, BERT and RoBERTa are still deterministic models that tend to make highly confident
predictions for ambiguous or unseen examples. As Guo et al. (2017) pointed out, modern neural networks are
frequently poorly calibrated, i.e. prediction confidence. Consequently, classification accuracy alone is insufficient
for applications such as fake news detection, where unreliable predictions may have significant societal
consequences.

To overcome such limitation, recent work has studied uncertainty estimation methods that quantify the confidence of
the model and recognise predictions that need further verification. Bayesian Neural Networks provide principled
uncertainty estimates, but they are computationally expensive and difficult to scale. This is a alternative called
Monte Carlo (MC) Dropout (Gal and Ghahramani, 2016), where we do multiple stochastic forward passes during
inference with dropout layers activated, and obtain an efficient approximation to Bayesian inference. The variation
between repeated predictions allows the model to estimate predictive uncertainty without requiring major
modifications to the underlying neural network. Despite the wide use of MC Dropout in domains such as medical
image analysis and autonomous systems, its application in the fake news detection is relatively limited.
Incorporating uncertainty estimation therefore provides an opportunity to improve the reliability of misinformation
detection by distinguishing confident predictions from those requiring additional analysis.

With the development of classification models, researchers have increasingly recognised the limitations of
content-only fake news detection systems. These approaches determine whether an article is trustworthy based only
on the text, without external validation. They are doomed to fail when the truthfulness of the claim needs world
knowledge that was not observed at training time. To overcome this limitation, retrieval-based fact checking
involves retrieving relevant evidence from trusted external sources before validating a claim. At the heart of these
systems lies Natural Language Inference (NLI), which decides if retrieved evidence supports, contradicts or is
unrelated to a given claim. NLI has become a central benchmark for semantic reasoning, and has heavily influenced
the design of modern inference models, thanks to the Stanford Natural Language Inference (SNLI) corpus (Bowman
et al., 2015). More recently, DeBERTa introduced disentangled attention and enhanced positional encoding, enabling
improved modelling of semantic relationships and achieving state-of-the-art performance on several language
understanding benchmarks (He et al., 2021). These capabilities make DeBERTa particularly suitable for evidence
verification, where understanding nuanced relationships between claims and supporting evidence is essential.
The robustness of fake news detection systems is also influenced by the choice of evaluation datasets. The
WELFake dataset contains more than 72,000 labelled news articles and thus is a large benchmark for
document-level classification (Verma et al., 2021). The LIAR dataset (Wang, 2017) consists of short political
statements with labels over more than two kinds of truthfulness, posing a more challenging claim-level verification
task. Model evaluation on both datasets allows for evaluation over different types of misinformation, as well as
increases confidence in model generalisation.

Each of these techniques such as transformer-based classification, uncertainty estimation and retrieval-based
verification have shown promising results, but relatively few studies have been conducted on the integration of these
techniques in a single framework. The existing approaches mainly focus on the classification accuracy, but ignore
the prediction reliability or do not selectively verify the uncertain outputs. This dissertation fills this research gap by
proposing a unified framework combining RoBERTa for initial fake news classification, Monte Carlo Dropout for
uncertainty estimation and DeBERTa based Natural Language Inference for evidence verification. We aim to
improve reliability and transparency of automated fake news detection by verifying uncertain predictions only and
reducing the computational overhead of the verification process.

## 3. Proposed Method
Describe the method(s) you are proposing, developing, or using, i.e., details of the
algorithms may be included here.

### 3.2 Problem Formulation

𝑁
Fake news detection is treated as a binary document classification. Given a dataset 𝐷 = {(𝑥𝑖, 𝑦𝑖)}𝑖 = 1 with

𝑦𝑖 ϵ {0, 1) with (real, fake), each document 𝑥𝑖 is formed by concatenating an article’s title and body text. The

objective is to learn 𝑝(𝑦 | 𝑥), a distribution over the two classes, such that the model generalises to unseen articles
rather than memorising surface patterns of the training set.

### 3.3 Classifier: RoBERTa with Low-Rank Adaptation (LoRA)

The base classifier is a pretrained RoBERTa encoder 𝑓θ, which maps a tokenised document to a pooled
𝑑
representation of ℎ = 𝑓θ(𝑥) ϵ 𝑅 . A classification head g projects the representation to class logits.

```text
                      z = gϕ(h) ∈ Rᶜ,        p(y = c | x) = exp(z_c) / Σᶜ_{c′=1} exp(z_c′)

```

The model is trained by minimising cross- entropy loss:

```text
L = -(1/N) Σᴺ_{i=1} Σᶜ_{c=1} 1[y_i = c] log p(y = c | x_i)

```

#### Choosing a model?

Encoders were selected over other normal linear classification models as they have the capabilities of learning
contextual, semantic representations of text. While systematic regression only sees word frequency counts with no
sense of context, order and meaning. Giving encoders the detection of subtle phrasing that a bag of words model
cannot.

RoBERTa (Liu et al., 2019) was chosen out of the encoders as it retains BERT’s architecture however it was trained
on 10x the amount of data as well as through the removal of NSP (next-sentence-prediction objective) that did not
benefit the model and inherently inhibited the model. This ultimately allows a stronger downstream performance
with no changes and no additional implementation cost.

Low-Rank Adaptation (LoRA). Is used to freeze the pretrained weights and learn a low rank update:

```text
W = W₀ + (a/r)BA,    B ∈ R^(d×r), A ∈ R^(r×k), r ≪ min(d, k)

```

The classification head as well as the parameters A and B are updated during training leaving Wo frozen. This is
applied to the query and value projection matrices within each self attention block. The reason behind it was for
parameter efficiency. Using this allows the use for 0.236% of the total parameters.

Total parameters: 124,942,082
Trainable parameters (After LoRA): 294,912

This inherently reduces the risk of overfitting on a moderately sized dataset (WELFake) whilst removing the cost of
memory and speeding up the training time.

### 3.4 Monte Carlo Dropout for Uncertainty Quantification

A point prediction 𝑦 alone does not indicate how much the classifier’s decision should be trusted. We estimate
predicted uncertainty using Monte Carlo (MC) Dropout (Gal & Ghahramani, 2016): dropout layers usually being
disabled but instead are kept active and the same input is passed through the network T times, each pass using an
independently sampled dropout mask. Which yields T stochastic predictive distributions 𝑝𝑡(𝑦 | 𝑥), which are used as

an approximation to Bayesian models averaging over the network parameters.

#### Mean prediction and confidence

```text
p̄(y | x) = (1/T) Σᵀ_{t=1} p_t(y | x),    ŷ = argmax p̄(y = c | x),    conf(x) = max p̄(y = c | x)

```

Uncertainty decomposition. Total predictive uncertainty is decomposed into an aleatoric component (chance
component) and an epistemic component (uncertainty caused by the models own lack of confidence)

```text
H[p̄] = -Σᶜ_{c=1} p̄(y = c | x) log p̄(y = c | x)    (total predictive entropy)

(1/T) Σᵀ_{t=1} (-Σᶜ_{c=1} p_t(y = c | x) log p_t(y = c | x))

```

**Expected entropy (aleatoric)**

```text
J = H[p̄] - (1/T) Σᵀ_{t=1} H[p_t]

```

**Mutual information (epistemic)**

A high epistemic value indicates the T stochastic passes disagreed with one another. Using this method we are able
to depict if the network is under modelled (The transformer has not fully learnt the pattern). Or , inherently
ambiguous (where the model isn't certain but the entropy is just over 0.5 meaning that it is not inherently confident
in its decision) This also allows for the evaluation of the models stability as well as further clarifying the
effectiveness of the model.

The live decision pipeline (pipeline.py) uses a lighterweight measure for the same T =30 passes: the standard
deviation of the fake-class probability across passes, σ𝑓𝑎𝑘𝑒 = 𝑠𝑡𝑑 ( 𝑝𝑡 ( 𝑦 = 𝑓𝑎𝑘𝑒 ∣ 𝑥 )). This is mapped to a

plain-language stability label for the end user - Very stable (σ𝑓𝑎𝑘𝑒<0.02), Stable (<0.05), Somewhat unstable (<0.10),

or Unstable (otherwise) via describe_mc_stability().

### 3.5 Verification Tier: Retrieval and Natural Language Inference

A second signal, structured around the FEVER paradigm (Thorne et al., 2018), checks the classifier’s stylistic
judgement against retrieved evidence. Every input runs through both the classifier and this tier unconditionally and
in parallel there is no confidence gate and the classifier label is always final; the verdict is reported as supporting
context only (“The classifier label is final. Verification is supporting context only.”). The article’s headline is used as
the claim c; the top-k snippets from a live web search (retrieve_evidence, no API key) form 𝐸𝑘 , each scored by a

pretrained FEVER-tuned DeBERTa NLI model (nli_scores) against c:

```text
s(e, c) = softmax(NLI(e, c)) ∈ Δ²,    s(e, c) = (s_ent, s_neu, s_con)

```

### 3.6 Explanation Agent

A final stage (explain.py, Claude Haiku 4.5, temperature =0) narrates each decision for a human reader from
structured signals only label, confidence, MC uncertainty, verdict, entailment/contradiction scores, evidence count
never raw article text, so it cannot invent unsupported content. Its system prompt fixes the classifier label as
non-negotiable: conflicting NLI evidence is described as context, never as grounds to reclassify. Output is limited to
two or three plain-English sentences

### 3.7 Baseline: TF-IDF with Logistic Regression

Included not to compete with RoBERTa + LoRA (Verma et al.) but to justify the pipeline as well as using it as a
bench mark because these models are used in conjunction in real life deployed fake news models. If the RoBERTa +
LoRA could not clearly outperform it, the proposed model would not be earning its keep.

```text
ϕ_j(x) = tf_j(x) log(N/df_j),    p(y = 1 | x) = σ(wᵀϕ(x) + b)

L_base = -(1/N) Σᴺ_{i=1} [y_i log p_i + (1-y_i) log(1-p_i)] + λ‖w‖²₂

```

## 4. Implementation

### 4.1 Dataset and Preprocessing

The classifier is trained on WELFake instead of LIAR because LIAR only contains 12.8k short especially that the
encoder RoBERTa having over a million parameters (around 120k when LoRA is applied), single sentence political
statements, while WELFake provides around 72k full news articles with both title and body which provides a closer
match to the output that we are looking for. The dataset is also large enough to properly fine-tune a transformer like
RoBERTa. Cleaning removed 558 rows with missing titles (filled instead), 39 with missing text, and 8,618 exact
duplicates, leaving 62,649 rows, with title and text concatenated into a single content field. A stratified 70/15/15
split (random_state=42) yields 43,854 training, 9,397 validation, and 9,398 test rows.

### 4.2 Model and Training Configuration

roberta-base (AutoModelForSequenceClassification, num_labels=2), tokenised with max_length=256,
truncation=True. LoRA configuration (peft):

| Hyperparameter | Value |
|---|---|
| Rank r | 8 |
| Scaling a | 16 |
| Target modules | query, value |
| Dropout | 0.05 |
| Bias | none |
| Task type | SEQ_CLS |

This yields 294,912 trainable parameters out of 124,942,082 (0.2360%). Training uses AdamW (lr =2×10-4 , weight
decay 0.01, batch size 32, no scheduler) for up to 3 epochs with early stopping (patience =1 on validation loss): loss
was 0.0290, 0.0119, 0.0139 across epochs 1 - 3, triggering a stop and restoring the epoch-2 adapter state
(get_peft_model_state_dict/set_peft_model_state_dict). The LoRA adapter is saved via save_pretrained; the
classification head is saved separately (classifier_head.pt) since it is newly initialised, not part of the frozen base or
adapter weights.

### 4.3 Monte Carlo Dropout Implementation

An enable_dropout() helper re-enables only torch.nn.Dropout layers after .eval(). T =30 forward passes are run
under torch.no_grad() to compute mean prediction , entropy, and mutual information (§3.4), after which the model is
reset to .eval() to avoid dropout remaining active unintentionally.

### 4.4 Verification Implementation
Implemented across five stages in verify.py:

| Stage | Function | Description |
|---|---|---|
| Trigger | run_pipline | Runs unconditionally for every input in parallel with classification |
| Claim | verify_claim | Article headline used as claim |
| Retrieve | retrieve_evidence | Live web search top-k = 5 snippets; no API needed |
| NLI | nli_scores | Zero-shot using FEVER-tuned DeBERTa |
| Aggregate | verify_claim | Max score per category |
NLI encoding uses its own tokenizer (max_length=512), separate from the classifier’s tokenizer settings in 4.2.

Configuration: k =5, γ =0.6, checkpoint MoritzLaurer/DeBERTa-v3-base-mnli-feveranli.

### 4.5 Explanation Agent Implementation

explain_decision() (explain.py) validates seven structured arguments, serialises them to JSON, and sends them to
claude-haiku-4-5 (temperature =0, max_tokens=220) with a system prompt fixing the classifier label as final.
Missing or rejected API keys raise distinct exceptions (MissingAnthropicKeyError, AnthropicKeyRejectedError)
rather than failing silently.

### 4.6 Reproducibility and Environment

- Platform: Google Colab, T4 GPU (torch.cuda.is_available()).

- Libraries: PyTorch, Hugging Face transformers, peft, datasets, scikit-learn.

- Seeds: random_state=42 fixes the data split.

- Code: cleaning, training, and MC Dropout inference implemented in clean_welfake.py, train_welfake_pytorch.py,
and mc_dropout_uncertainty.py.

## 5. Experiments

### 5.1 Dataset

WELFake (Kaggle saurabhshahane/fake-news-classification), combining four existing collections (Kaggle,
McIntire, Reuters, BuzzFeed Political): 72,134 raw articles (37,106 fake, 35,028 real). After cleaning (558 missing
titles filled, 39 missing-text rows dropped, 8,618 duplicates removed), 62,649 rows remain, split 70/15/15 (43,854
train / 9,397 validation / 9,398 test, random_state=42).

### 5.2 Software

PyTorch (PyTorch Contributors); Hugging Face transformers for RoBERTa (HuggingFace); peft for LoRA
fine-tuning (LoRA); datasets for data loading; scikit-learn (Scikit-learn) for the TF-IDF + logistic regression
baseline and evaluation metrics.

### 5.3 Hardware

Google Colab, single NVIDIA T4 GPU.

## Results and Discussion

The classifier reached near-ceiling accuracy within WELFake [Missing bibliography entry: verma2021welfake], but
we do not treat that score as cross-dataset evidence. We evaluated RoBERTa-LoRA on the deterministic, stratified
70/15/15 split with seed 42 rather than drawing a fresh random split, preserving class balance and reproducibility.
The held-out test set contained 9,398 unseen articles, comprising 5,193 real and 4,205 fake articles. We report
accuracy, per-class precision, recall and F1, with unweighted macro averages. Table 1 gives the scores that the error
analysis then explains.

| Model or class | Accuracy | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|---:|
| RoBERTa-LoRA overall | 0.9957 | 0.9958 | 0.9956 | 0.9957 | 9,398 |
| Real | | 0.9952 | 0.9971 | 0.9962 | 5,193 |
| Fake | | 0.9964 | 0.9941 | 0.9952 | 4,205 |
| TF-IDF + logistic regression | 0.9526 | | | 0.9521 | |
| Majority-class baseline | 0.5526 | | | 0.3559 | |

Table 1. Held-out WELFake performance and baseline comparison.

The 40 errors show why near-ceiling performance still needs a dataset boundary. We use true labels as rows and
predicted labels as columns in real/fake order. The matrix below exposes that error pattern.

| | Predicted real | Predicted fake |
|---|---:|---:|
| True real | 5178 | 15 |
| True fake | 25 | 4180 |
The classifier labelled 15 real articles as fake and 25 fake articles as real. This pattern agrees with the lower
fake-class recall of 0.9941 compared with the real-class recall of 0.9971. RoBERTa-LoRA exceeded TF-IDF by 4.31
percentage points in accuracy and 4.36 points in macro F1. This margin supports contextual representations over
word-frequency features on this split, but it does not establish the same advantage on LIAR or under distribution
shift. Near-ceiling WELFake accuracy may reflect source or style shortcut learning rather than transferable veracity
cues, a known failure mode in which a model performs well on a benchmark but fails under harder transfer
conditions (Geirhos et al., 2020). We next test whether MC Dropout identifies the few cases that remain difficult.

MC Dropout left accuracy effectively unchanged, while entropy separation supplied the operative result. Across 30
stochastic passes, MC Dropout [Missing bibliography entry: gal2016dropout] reached 0.9967 accuracy compared
with 0.9957 under deterministic inference, a difference of 0.10 percentage points. Its ECE [Missing bibliography
entry: guo2017calibration] was 0.0055 compared with 0.0028 for deterministic confidence, so MC averaging did not
improve calibration. Mean predictive entropy was 0.0333 for correct predictions and 0.4816 for incorrect
predictions. On the notebook's rejection curve under MC labels, deferring the 100 highest-entropy cases left 9,298
articles at 98.94% coverage and raised retained accuracy to 99.88% (9,287/9,298). The useful finding is therefore
error separation for routing, not a material accuracy gain or better calibration. That routing result leads to the test of
whether verification can rescue the deferred cases.

We expected external evidence to rescue uncertain predictions, but direct NLI override produced the opposite result.
We selected the 100 lowest-confidence test articles rather than a random 100 because the experiment targeted the
cases identified for intervention, not average behaviour across all 9,398 articles [Missing bibliography entry:
geifman2017selective]. We tested direct override rather than the deployed advisory use of NLI so that the evaluation
could measure whether external evidence corrected classifier errors. The rule mapped supported to real and refuted
to fake, while insufficient retained the classifier label. We then applied DDG retrieval and DeBERTa NLI. Table 2
records how that expectation failed.

| Verdict | Rows | Classifier accuracy | Overrides | Fixed | Broken | Final accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Supported | 42 | 73.8% (31/42) | 14 | 7 | 7 | 73.8% (31/42) |
| Refuted | 32 | 71.9% (23/32) | 20 | 4 | 16 | 34.4% (11/32) |
| Insufficient | 26 | 84.6% (22/26) | 0 | 0 | 0 | 84.6% (22/26) |
| Overall | 100 | 76.0% (76/100) | 34 | 11 | 23 | 64.0% (64/100) |

Table 2. Escalation outcomes on the 100 lowest-confidence articles.

The override moved accuracy in a consistently harmful direction, although this sample does not establish a
statistically significant difference. Baseline classifier accuracy was 76.0% (76/100; 95% Wilson CI, 66.8% to
83.3%). Post-override accuracy was 64.0% (64/100; 95% Wilson CI, 54.2% to 72.7%). Overrides fixed 11 errors
and broke 23 correct predictions; the two-sided exact McNemar test gave p = 0.058, a direction consistent with harm
that does not reach the conventional 0.05 threshold in this paired sample of n = 100. The supported group was
neutral with seven fixes and seven breaks. The refuted group caused the main failure, with four fixes and 16 breaks
and a fall from 71.9% to 34.4%. The insufficient group retained its labels and achieved the best classifier-only
accuracy of 84.6%. We therefore reject the current unthresholded override. A contradiction score from a live snippet
does not establish that a complete article is false, so external evidence should support human review or a policy with
validated evidence-quality thresholds. This failure makes retrieval coverage and source stability the next question.

Google Fact Check supplied no coverage on the fixed bucket, while its live fallback was unstable. The API
produced 0/100 coverage. Every row recorded no Google fact-check match, with 100/100 clean no-match responses
and 0/100 request or key errors. This result is consistent with a task mismatch because Google Fact Check uses
claim-level indexing, whereas our unit of evaluation was a full article. The DDG fallback verdict distribution shifted
from 42/32/26 supported, refuted, and insufficient verdicts to 30/12/58 on the repeated run, while both runs retained
64.0% accuracy. The unchanged aggregate concealed live-retrieval instability. Coverage alone does not show
whether the deployed output remained faithful to its inputs, which we test next.

The deployed pipeline ran, but its explanation layer remained fallible. We verified the public ZeroGPU Space on 20
July 2026 through its /analyze endpoint. The call completed in 64.4 seconds and the classifier returned REAL at
99.57% confidence, "Very stable" prediction stability, and an NLI verdict of Supported. This runtime included live
DDG retrieval and two ZeroGPU allocations. Explanation faithfulness was 80.0% (8/10; 95% Wilson CI, 49.0% to
94.3%). This was a single-reviewer assessment rather than a population estimate. Row 8 reversed a supported NLI
verdict, while row 94 attributed the classifier decision to retrieved evidence that the classifier never received. These
operating and faithfulness results complete the evidence needed to judge the proposal criteria.

Against the proposal's five success criteria, the system met four. MET, classifier quality. Macro F1 reached 0.9957
on held-out WELFake [Missing bibliography entry: verma2021welfake], which met the proposal's
published-baseline target but did not establish LIAR performance. NOT MET, calibration. MC ECE was 0.0055
rather than below deterministic ECE of 0.0028 [Missing bibliography entry: guo2017calibration]. MET, selective
deferral. Retained accuracy rose when we deferred the 100 highest-entropy predictions [Missing bibliography entry:
geifman2017selective]. MET, verification measurement. Table 2 records coverage and classifier-verifier
disagreement, while Google Fact Check returned 0/100 coverage. MET, explanation faithfulness. Eight of 10
sampled explanations matched their structured inputs. The unmet calibration criterion and the evidence-quality
failures lead directly to the study limits.

Four limits bound these results. First, the classifier metrics cover one held-out WELFake split rather than LIAR or
another external corpus. Second, escalation covers 100 of 9,398 test articles and therefore measures behaviour only
in the lowest-confidence bucket. Third, live DDG results can change between runs. Fourth, the explanation review
used 10 examples and one reviewer, which leaves a wide 49.0% to 94.3% Wilson interval and provides no
multi-reviewer agreement measure. These boundaries constrain the conclusion and define the next evaluation steps.

## Conclusions

The classifier met its in-dataset target, but the evidence layer did not earn authority over its labels. We built and
publicly deployed the proposal's three-layer detection, verification, and explanation system using RoBERTa-LoRA
on WELFake with DDG and DeBERTa NLI. MC Dropout supplied a useful routing signal through entropy
separation, not a material accuracy or calibration gain. The escalation experiment then found a consistent direction
of harm from direct override, although the sample was too small to cross the 0.05 significance threshold. This result
extends FEVER-style retrieval and entailment work by showing that a live verifier can reduce end-to-end accuracy
when its verdict directly replaces an article label (Thorne et al., 2018; Hanselowski et al., 2018; Soleimani et al.,
2020). Google Fact Check's claim-level index did not cover the article bucket, and repeated DDG retrieval changed
verdicts. We therefore keep the classifier label final and present retrieved evidence as a separate signal for review.
Cross-dataset testing, validated routing thresholds, stable evidence snapshots, and a larger multi-reviewer
explanation study are the next tests of whether this system generalises.

## Acknowledgements
List acknowledgements if any. For example, if someone provided you a dataset, or you
used someone else’s resources, this is a good place to acknowledge the help or
support you received.

## Figures

| Symbol | Meaning |
|---|---|
| x, y | input document; label (0 = real, 1 = fake) |
| N, C, L | number of examples; number of classes (= 2); tokenised sequence length |
| f, g | RoBERTa encoder; classification head |
| h, z, p | pooled representation; logits; softmax probabilities |
| W₀, B, A, r, a | frozen pretrained weight; LoRA factor matrices; rank; scaling factor |
| T, p_t, p̄ | number of MC Dropout passes; per-pass predictive distribution; mean predictive distribution |
| H[p̄], J | predictive (total) entropy; mutual information (epistemic uncertainty) |
| t, γ | confidence threshold for routing to verification; NLI verdict threshold |
| E_k | passage embedder; cosine similarity; top-k retrieved evidence set |
| ϕ(x), V, w, λ | TF-IDF feature vector; vocabulary; logistic regression weights; L2 regularisation strength |

## Contributions
Describe the contributions of each team member who worked on this project.
Note: For rubric, refer to the Group Project Guidelines:
https://qmplus.qmul.ac.uk/mod/resource/view.php?id=3794535

## References

Thorne, James, et al. “Fever: A Large-Scale Dataset for Fact Extraction and VERification.”
Association for Computational Linguistics, 1 June 2018.

Gal, Yarin, and Zoubin Ghahramani. “Dropout as a Bayesian Approximation: Representing
Model Uncertainty in Deep Learning.” PMLR, 11 June 2016.

Hu, Edward J., et al. “LoRA: Low-Rank Adaptation of Large Language Models.”
arXiv:2106.09685 [Cs], 16 Oct. 2021, https://arxiv.org/abs/2106.09685. Accessed 21 July 2026.

Gal, Yarin, and Zoubin Ghahramani. “Dropout as a Bayesian Approximation: Representing
Model Uncertainty in Deep Learning.” PMLR, 11 June 2016.

Verma, Pawan Kumar, et al. “WELFake: Word Embedding over Linguistic Features for Fake
News Detection.” IEEE Transactions on Computational Social Systems, vol. 8, no. 4, Aug. 2021,
pp. 881–893, 10.1109/tcss.2021.3068519.

Bowman, S., Angeli, G., Potts, C. and Manning, C.D. (2015). A large annotated corpus for
learning natural language inference. [online] aclanthology.org, pp.632–642.
doi:10.18653/v1/D15-1075.

Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019). BERT: Pre-training of Deep
Bidirectional Transformers for Language Understanding. Proceedings of the 2019 Conference of
the North, [online] 1, pp.4171–4186. doi:10.18653/v1/n19-1423.
Mimura, M. and Ishimaru, T. (2024). Analyzing common lexical features of fake news using
multi-head attention weights. Internet of Things, [online] 28, p.101409.
doi:10.1016/j.iot.2024.101409.

Guo, C., Pleiss, G., Sun, Y. and Weinberger, K.Q. (2017) On Calibration of Modern Neural
Networks. In: Proceedings of the 34th International Conference on Machine Learning
(ICML 2017). Sydney: PMLR, pp. 1321–1330.

He, P., Liu, X., Gao, J. and Chen, W. (2021) DeBERTa: Decoding-enhanced BERT with
Disentangled Attention. In: International Conference on Learning Representations (ICLR
2021).

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M.S., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L.
and Stoyanov, V. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv
(Cornell University), 1. doi:10.48550/arxiv.1907.11692.

Shu, K., Sliva, A., Wang, S., Tang, J. and Liu, H. (2017). Fake news detection on social media: A
data mining perspective. ACM SIGKDD Explorations Newsletter, [online] 19(1), pp.22–36.
doi:10.1145/3137597.3137600.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł. and
Polosukhin, I. (2017) Attention Is All You Need. In: Advances in Neural Information
Processing Systems, Vol. 30. Red Hook, NY: Curran Associates, pp. 5998–6008.

Verma, P.K., Agrawal, P., Amorim, I. and Prodan, R. (2021). WELFake: Word Embedding Over
Linguistic Features for Fake News Detection. IEEE Transactions on Computational Social
Systems, 8(4), pp.881–893. doi:10.1109/tcss.2021.3068519.

Vosoughi, S., Roy, D. and Aral, S. (2018). The Spread of True and False News Online. Science,
359(6380), pp.1146–1151. doi:10.1126/science.aap9559.
Wang, W.Y. (2017). “Liar, Liar Pants on Fire”: A New Benchmark Dataset for Fake News
Detection. [online] ACLWeb, Vancouver, Canada: Association for Computational Linguistics,
pp.422–426. doi:10.18653/v1/P17-2067.

LoRA (2026) https://huggingface.co/docs/peft/en/package_reference/lora

Huggingface.Co, 2026,

https://huggingface.co/docs/transformers/v5.14.0/en/model_doc/auto#transformers.autoto

kenizer.

PyTorch Contributors. “PyTorch Documentation.” Pytorch.Org, 2023,

https://docs.pytorch.org/docs/2.13/index.html. Accessed 23 July 2026.

Scikit-learn. “Scikit-Learn: Machine Learning in Python.” 2024.
