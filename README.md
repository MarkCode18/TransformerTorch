# TransformerTorch 🚀
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-red?logo=pytorch)
![Tokenizers](https://img.shields.io/badge/Tokenizers-0.22.1-lightgrey?logo=huggingface)
![Demo](https://img.shields.io/badge/🤗-HuggingFace%20Demo-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)

Welcome to TransformerTorch! This repository contains three main parts:

## ⚡ **Transformer Implementation**
A Jupyter Notebook [TransformerTorch.ipynb](https://github.com/HooM4N/TransformerTorch/blob/main/TransformerTorch.ipynb) with a complete from‑scratch implementation of the **Transformer architecture** based on the paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).

Implemented modules: `ScaledDotProductAttention`, `MultiHeadAttention`, `EncoderLayer`, `DecoderLayer`, `TransformerEncoder`, `TransformerDecoder`, `Transformer` (encoder + decoder), and a custom `PositionalEmbedding` (implemented instead of the sinusoidal embedding in the original paper).

## 🌐 **Training a Neural Machine Translation System with Transformer**
Also included in the notebook: the Transformer architecture is applied to train a **Neural Machine Translation** system on 220K English–Spanish sentence pairs using a BPE tokenizer.

To improve training speed, performance, and efficiency, techniques such as *Mixed Precision*, *Weight Tying*, and *Shared Embeddings/Vocabulary across source and target languages* are used.

For inference, a custom batch-inference `greedy_decode` function was developed that computes encoder memory once and reuses it across all decoder timesteps, making translation faster and more memory‑efficient.

This project builds on my earlier RNN‑Attention based NMT work in [AttentionNMT](https://github.com/Hoom4n/AttentionNMT), where you can find the full details of dataset sourcing, preparation and preprocessing, tokenizer training, and the implementation of custom PyTorch data modules.

## 🎛️ **NMT System Demo**
A Gradio‑based demo (`app.py`) showcases the translation system. Inference modules are located in `src/`. You can try the NMT system using the following options:

#### 🌐 Online Demo
Available on Hugging Face Spaces: [https://hoom4n-transformertorch.hf.space/](https://hoom4n-transformertorch.hf.space/)

#### 🐳 Run with Docker
```bash
git clone https://github.com/hoom4n/TransformerTorch.git
cd TransformerTorch

docker compose up --build   # first run
docker compose up           # subsequent runs
```

#### 💻 Run Locally
```bash
git clone https://github.com/hoom4n/TransformerTorch.git
cd TransformerTorch

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python app.py
```
