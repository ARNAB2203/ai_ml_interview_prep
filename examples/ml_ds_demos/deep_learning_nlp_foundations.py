"""Deep-learning and NLP fundamentals: autograd, architectures, attention, TF-IDF.

Run after installing examples/requirements-ml-ds.txt:
    python examples/ml_ds_demos/deep_learning_nlp_foundations.py

Every input is synthetic or local. The script is an architecture and shape
reasoning exercise—not a claim that small randomly initialized networks solve a
real task.
"""

# %% Imports
from __future__ import annotations

from math import sqrt

import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# %% Step 1 — forward/backward pass and a finite-difference gradient check
def gradient_check() -> dict[str, float]:
    feature = torch.tensor([2.0])
    target = torch.tensor([5.0])
    weight = torch.tensor([1.5], requires_grad=True)
    loss = torch.mean((weight * feature - target) ** 2)
    loss.backward()
    autodiff_gradient = float(weight.grad.item())

    epsilon = 1e-3
    with torch.no_grad():
        plus = torch.mean(((weight + epsilon) * feature - target) ** 2)
        minus = torch.mean(((weight - epsilon) * feature - target) ** 2)
    numerical_gradient = float(((plus - minus) / (2 * epsilon)).item())
    return {"loss": float(loss.item()), "autodiff": autodiff_gradient, "finite_difference": numerical_gradient}


# %% Step 2 — CNN, RNN, LSTM, and GRU shape contracts
def architecture_shapes() -> dict[str, tuple[int, ...]]:
    torch.manual_seed(11)
    sequence = torch.randn(3, 6, 8)  # batch, time, feature
    cnn = nn.Conv1d(in_channels=8, out_channels=12, kernel_size=3, padding=1)
    rnn = nn.RNN(input_size=8, hidden_size=10, batch_first=True)
    lstm = nn.LSTM(input_size=8, hidden_size=10, batch_first=True)
    gru = nn.GRU(input_size=8, hidden_size=10, batch_first=True)

    cnn_output = cnn(sequence.transpose(1, 2))  # Conv1d uses batch, channel, time
    rnn_output, _ = rnn(sequence)
    lstm_output, _ = lstm(sequence)
    gru_output, _ = gru(sequence)
    return {
        "cnn": tuple(cnn_output.shape),
        "rnn": tuple(rnn_output.shape),
        "lstm": tuple(lstm_output.shape),
        "gru": tuple(gru_output.shape),
    }


# %% Step 3 — scaled dot-product attention with a causal mask
def scaled_dot_product_attention() -> dict[str, tuple[int, ...] | float]:
    torch.manual_seed(13)
    batch, tokens, width = 2, 4, 8
    query = torch.randn(batch, tokens, width)
    key = torch.randn(batch, tokens, width)
    value = torch.randn(batch, tokens, width)
    scores = query @ key.transpose(-2, -1) / sqrt(width)
    causal_mask = torch.triu(torch.ones(tokens, tokens, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = weights @ value
    forbidden_mass = float(weights.masked_select(causal_mask.expand_as(weights)).sum().item())
    return {"weights": tuple(weights.shape), "output": tuple(output.shape), "masked_attention_mass": forbidden_mass}


# %% Step 4 — classical TF-IDF baseline before a transformer
def tfidf_baseline() -> dict[str, float]:
    corpus = [
        "deployment requires staging evaluation before canary rollout",
        "tool execution requires least privilege and audit logging",
        "human approval is required for regulated release",
    ]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    query = vectorizer.transform(["staging before canary deployment"])
    scores = cosine_similarity(query, matrix).ravel()
    return {"best_document_index": float(scores.argmax()), "best_similarity": float(scores.max())}


def demo() -> None:
    gradients = gradient_check()
    shapes = architecture_shapes()
    attention = scaled_dot_product_attention()
    tfidf = tfidf_baseline()
    print("Gradient check:", gradients)
    print("Architecture shapes:", shapes)
    print("Masked attention:", attention)
    print("TF-IDF baseline:", tfidf)
    assert abs(gradients["autodiff"] - gradients["finite_difference"]) < 0.02
    assert shapes["cnn"] == (3, 12, 6)
    assert shapes["lstm"] == (3, 6, 10)
    assert attention["output"] == (2, 4, 8)
    assert attention["masked_attention_mass"] < 1e-6
    assert tfidf["best_document_index"] == 0.0
    print("PASS: deep-learning shape, gradient, attention, and NLP-baseline checks completed.")


if __name__ == "__main__":
    demo()
