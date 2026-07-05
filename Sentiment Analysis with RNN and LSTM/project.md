# Sentiment Analysis with RNN & LSTM

A PyTorch implementation of binary sentiment classification on the IMDB movie review dataset, comparing a vanilla RNN against an LSTM to demonstrate the vanishing-gradient problem in practice.

## Overview

This project builds a sentiment classifier from scratch — custom tokenization, vocabulary construction, and a recurrent neural network — trained on the [IMDB Large Movie Review Dataset](https://huggingface.co/datasets/imdb) (25,000 train / 25,000 test reviews, balanced classes).

The core focus isn't just "build a classifier" but **why architecture choice matters**: an initial vanilla RNN implementation plateaued at ~58% accuracy (barely above random guessing), and diagnosing that failure led to a working LSTM at ~82% accuracy.

## Dataset

| Split | Reviews | Classes |
|---|---|---|
| Train | 25,000 | Positive / Negative |
| Test | 25,000 | Positive / Negative |

Reviews are raw HTML-scraped IMDB text (including `<br />` tags), variable length, English only.

## Pipeline

1. **Tokenization** — lowercase, strip non-alphanumeric characters, whitespace split
2. **Vocabulary** — built from training set token frequencies, `MIN_FREQ=2`, with `<PAD>` and `<UNK>` special tokens (final vocab size: 57,469)
3. **Encoding & padding** — sequences truncated/padded to `MAX_LEN=200`, true lengths tracked for `pack_padded_sequence`
4. **Model** — embedding layer → recurrent layer (RNN or LSTM) → linear classifier on the final hidden state
5. **Training** — Adam optimizer, cross-entropy loss, 5 epochs, batch size 32

## Architecture Comparison

| | Vanilla RNN | LSTM |
|---|---|---|
| Recurrent layer | `nn.RNN` | `nn.LSTM` |
| Test Accuracy | **57.84%** | **~82%** |
| Training loss behavior | Plateaued near `ln(2) ≈ 0.69`, unstable across epochs | Decreased steadily |
| Long-range dependency handling | Poor — gradient vanishes across ~200 timesteps | Strong — gated cell state preserves gradient flow |

### Why the gap is so large

A vanilla RNN's hidden state update (`h_t = tanh(W_h·h_{t-1} + W_x·x_t + b)`) requires backpropagating through repeated multiplication by `W_h` at every timestep. Over a 200-token sequence, this shrinks the gradient signal exponentially, so the network effectively can't learn from anything beyond the last ~10-20 tokens — a serious limitation for sentiment, where negation and qualifiers ("I thought I'd love this, but...") often appear far from the words they modify.

An LSTM introduces a separate **cell state** updated additively through gates:

```
f_t = sigmoid(...)              # forget gate
i_t = sigmoid(...)              # input gate
c_t = f_t * c_{t-1} + i_t * c̃_t # additive update — no repeated multiplication
o_t = sigmoid(...)              # output gate
h_t = o_t * tanh(c_t)
```

Because the cell-state update is additive rather than a repeated matrix multiplication, gradients can flow across many timesteps largely unchanged (the "constant error carousel"), letting the model actually learn long-range dependencies.

## Repository Structure

```
├── Sentiment_Analysis_with_RNN_and_LSTM_extended.ipynb   # Full notebook: data, models, training, evaluation
├── sentiment_lstm.pt                                      # Saved LSTM weights (generated after running the notebook)
└── README.md
```

## Results & Analysis

The extended notebook includes:

- **Metrics** — precision, recall, F1-score, confusion matrix (not just accuracy)
- **Train/val loss curves** — to check for overfitting across epochs
- **Qualitative error analysis** — inspecting misclassified reviews to understand failure modes (sarcasm, mixed sentiment, subtle negation)
- **Custom inference demo** — a `predict_sentiment(text)` function tested on hand-written sentences, including a negation trap case
- **Model summary card** — architecture, parameter count, vocab size, final test accuracy

## Key Takeaways

- Vanilla RNNs are not a safe default for sequences longer than a few dozen tokens — the vanishing gradient problem isn't theoretical, it shows up directly as a stalled loss curve
- Gradient clipping (`clip_grad_norm_`) is cheap insurance against instability regardless of recurrent architecture
- Aggregate accuracy alone can mask *why* a model fails — confusion matrices and manual error inspection surfaced patterns a single accuracy number couldn't

## Possible Extensions

- Bidirectional LSTM/GRU to capture context from both directions
- Pretrained embeddings (GloVe/Word2Vec) instead of learned-from-scratch embeddings
- Attention mechanism over hidden states instead of using only the final timestep
- Compare against a Transformer-based baseline (e.g. DistilBERT) as an upper bound

## Requirements

```
torch
datasets
scikit-learn
matplotlib
```

## Usage

```bash
pip install torch datasets scikit-learn matplotlib
jupyter notebook Sentiment_Analysis_with_RNN_and_LSTM_extended.ipynb
```
