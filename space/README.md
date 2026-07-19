---
title: News Signal Review
emoji: 📰
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.20.0
python_version: 3.12.12
app_file: app.py
---

# News Signal Review

This ZeroGPU demo classifies a headline or article with RoBERTa-LoRA, reports MC Dropout stability, and shows DDG plus DeBERTa evidence as context that never overrides the classifier label.

The Claude explanation is optional. Expand the key section to provide your own Anthropic API key for one request; the key is password-masked, never added to the environment, and cleared after analysis. Classification, uncertainty, and evidence remain available without a key.

See the [GitHub repository](https://github.com/KP-365/Fake_news) for training details, evaluation results, limitations, source code, and full documentation.
