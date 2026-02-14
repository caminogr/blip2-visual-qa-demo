---
title: BLIP-2 Visual QA Demo
emoji: 🖼️
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# BLIP-2 Visual QA Demo

A visual question answering demo powered by [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b).

## 🚀 Live Demo

👉 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/camino-gr/blip2-visual-qa-demo)**

## Features

### 1. Image Captioning
Upload an image and get an automatic caption describing its contents.

### 2. Visual QA
Upload an image and ask a question about it — the model will answer based on visual understanding.

### 3. Image Comparison
Upload two images and get descriptions of each, along with highlighted differences.

## Model

This demo uses **BLIP-2** (Bootstrapping Language-Image Pre-training) with the OPT-2.7B language model backbone. BLIP-2 bridges the modality gap between vision and language using a lightweight Querying Transformer (Q-Former).

## Tech Stack

- **Model:** `Salesforce/blip2-opt-2.7b` via 🤗 Transformers
- **UI:** Gradio 5
- **Runtime:** Docker (Python 3.12)

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

## License

MIT
