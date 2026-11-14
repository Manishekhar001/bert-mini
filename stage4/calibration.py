"""Calibration, abstention threshold and out-of-domain scores for one softmax classifier."""
from __future__ import annotations

import numpy as np

from common.metrics import softmax


def fit_temperature(logits, labels) -> float:
    """Choose one positive temperature that minimizes validation negative log-likelihood."""
    from scipy.optimize import minimize_scalar

    logits, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels):
        raise ValueError("fit_temperature expects logits [reports, classes] and one label per report")
    if logits.shape[1] < 1:
        raise ValueError("fit_temperature needs at least one class")
    valid = labels >= 0
    z, y = logits[valid], labels[valid]
    if len(y) == 0:
        return 1.0
    if np.any(y >= logits.shape[1]):
        raise ValueError("A label is outside the logits' class range")

    def nll(log_temperature):
        probs = softmax(z / np.exp(log_temperature))
        return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean())

    fit = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(fit.x))


def calibrated_probs(logits, temperature: float = 1.0):
    """Convert logits into calibrated class probabilities with softmax(logits / temperature)."""
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be a finite positive number, got {temperature}")
    return softmax(np.asarray(logits, dtype=np.float64) / temperature)


def threshold_for_accuracy(confidence, correct, target: float = 0.9):
    """Lowest confidence cutoff whose accepted validation predictions reach the target accuracy."""
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidence.ndim != 1 or correct.ndim != 1 or len(confidence) != len(correct):
        raise ValueError("confidence and correct must be equally sized one-dimensional arrays")
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"target accuracy must be between 0 and 1, got {target}")
    if not len(confidence):
        return None, 0.0, float("nan")
    order = np.argsort(-confidence, kind="stable")
    conf, hits = confidence[order], correct[order]
    running_accuracy = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    end_of_ties = np.r_[conf[1:] != conf[:-1], True]
    possible = np.flatnonzero((running_accuracy >= target) & end_of_ties)
    if not len(possible):
        return None, 0.0, float(running_accuracy.max())
    index = int(possible.max())
    return float(conf[index]), float((index + 1) / len(conf)), float(running_accuracy[index])


def domain_score(kind: str, logits, probabilities):
    """One score per report: max probability or log-sum-exp of raw logits.

    For both choices higher is interpreted as more familiar/in-domain by this project's convention.
    """
    logits = np.asarray(logits, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if kind == "confidence":
        return probabilities.max(axis=1)
    if kind == "energy":
        maximum = logits.max(axis=1, keepdims=True)
        return (maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1)))
    raise ValueError(f"unknown domain score {kind!r}")
