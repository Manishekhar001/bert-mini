"""Metrics for the project's single 18-class prediction task. NumPy/SciPy only."""
from __future__ import annotations

import numpy as np


def softmax(x, axis: int = -1):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 0 or x.shape[axis] == 0:
        raise ValueError("softmax needs at least one value along its class axis")
    x = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=axis, keepdims=True)


def logits_to_probs(logits):
    return softmax(logits)


def reliability_bins(conf, hit, n_bins: int = 15):
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    conf, hit = np.asarray(conf, dtype=np.float64), np.asarray(hit, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    return [(float(conf[m].mean()), float(hit[m].mean()), int(m.sum()))
            for b in range(n_bins) if (m := idx == b).any()]


def ece(conf, hit, n_bins: int = 15) -> float:
    if len(conf) == 0:
        return float("nan")
    return float(sum(n / len(conf) * abs(acc - mean_conf)
                     for mean_conf, acc, n in reliability_bins(conf, hit, n_bins)))


def _f1(tp: float, fp: float, fn: float) -> float:
    denom = 2.0 * tp + fp + fn
    return 2.0 * tp / denom if denom else 0.0


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    """Mean class F1, omitting classes absent from the evaluated labels."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        if tp + fn:
            scores.append(_f1(tp, fp, fn))
    return float(np.mean(scores)) if scores else float("nan")


def evaluate(probs, labels) -> dict:
    probs, labels = np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or labels.ndim != 1 or len(probs) != len(labels):
        raise ValueError("evaluate expects a [reports, classes] probability matrix and one label per report")
    if probs.shape[1] < 1:
        raise ValueError("evaluate needs at least one class probability")
    valid = labels >= 0
    probs, labels = probs[valid], labels[valid]
    if len(labels) == 0:
        return {"n": 0, "accuracy": float("nan"), "macro_f1": float("nan"),
                "nll": float("nan"), "ece": float("nan")}
    if np.any(labels >= probs.shape[1]):
        raise ValueError("A label is outside the probability matrix's class range")
    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    nll = -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()
    return {
        "n": int(len(labels)),
        "accuracy": float((predictions == labels).mean()),
        "macro_f1": macro_f1(labels, predictions, probs.shape[1]),
        "nll": float(nll),
        "ece": ece(confidence, predictions == labels),
    }


def auroc(in_domain_scores, outside_scores) -> float:
    """Probability an in-domain score is higher than an outside score (ties count half)."""
    from scipy.stats import rankdata

    inside = np.asarray(in_domain_scores, dtype=np.float64)
    outside = np.asarray(outside_scores, dtype=np.float64)
    if len(inside) == 0 or len(outside) == 0:
        return float("nan")
    ranks = rankdata(np.concatenate([inside, outside]))
    n_in, n_out = len(inside), len(outside)
    return float((ranks[:n_in].sum() - n_in * (n_in + 1) / 2) / (n_in * n_out))
