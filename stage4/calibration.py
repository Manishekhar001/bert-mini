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


def apply_bias(logits, bias):
    """Add one offset per class to the logits. With bias=None the logits are returned unchanged."""
    logits = np.asarray(logits, dtype=np.float64)
    if bias is None:
        return logits
    bias = np.asarray(bias, dtype=np.float64)
    if bias.shape != (logits.shape[-1],):
        raise ValueError(f"class bias needs {logits.shape[-1]} values, got shape {bias.shape}")
    return logits + bias


def fit_class_bias(logits, labels, l2: float = 1e-2):
    """One additive offset per class that minimizes validation negative log-likelihood.

    Why: the training split is rebalanced but validation/test keep the natural class mix, so the model's built-in
    class frequencies are wrong for the data it is used on. A per-class offset fixes that without retraining.
    A small L2 penalty stops offsets of very rare classes from running away. Offsets are centred, because softmax
    ignores a shift that is the same for every class.
    """
    from scipy.optimize import minimize

    logits, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels):
        raise ValueError("fit_class_bias expects logits [reports, classes] and one label per report")
    valid = labels >= 0
    z, y = logits[valid], labels[valid]
    if len(y) == 0:
        return np.zeros(logits.shape[1])
    if np.any(y >= logits.shape[1]):
        raise ValueError("A label is outside the logits' class range")
    n, k = z.shape
    onehot = np.zeros((n, k))
    onehot[np.arange(n), y] = 1.0

    def objective(bias):
        probs = softmax(z + bias)
        nll = -np.log(np.clip(probs[np.arange(n), y], 1e-12, 1.0)).mean()
        grad = (probs - onehot).mean(axis=0) + 2.0 * l2 * bias
        return nll + l2 * float(bias @ bias), grad

    fit = minimize(objective, np.zeros(k), jac=True, method="L-BFGS-B")
    return fit.x - fit.x.mean()


def choose_bias_strength(logits, labels, strengths=(0.0, 0.25, 0.5, 0.75, 1.0), l2: float = 1e-2,
                         folds: int = 5, seed: int = 0):
    """Pick how much of the fitted bias to apply, using validation data only.

    Cross-validation: fit the bias on 4/5 of the validation reports, predict the other 1/5, pool all held-out
    predictions, and score 0.5 * accuracy + 0.5 * macro-F1. Strength 0 means no correction. Returns
    (best strength, one result dict per strength). Ties go to the smaller strength.
    """
    from common.metrics import macro_f1

    logits, labels = np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    valid = labels >= 0
    z, y = logits[valid], labels[valid]
    if len(y) < 2 * folds:
        return 0.0, []
    rng = np.random.default_rng(seed)
    fold_of = np.zeros(len(y), dtype=np.int64)
    offset = 0
    for c in np.unique(y):                       # spread every class evenly over the folds
        idx = rng.permutation(np.flatnonzero(y == c))
        fold_of[idx] = (np.arange(len(idx)) + offset) % folds
        offset += len(idx)
    biases = [fit_class_bias(z[fold_of != f], y[fold_of != f], l2) for f in range(folds)]
    results = []
    for strength in strengths:
        predicted = np.zeros(len(y), dtype=np.int64)
        for f in range(folds):
            held_out = fold_of == f
            predicted[held_out] = (z[held_out] + strength * biases[f]).argmax(axis=1)
        accuracy = float((predicted == y).mean())
        f1 = macro_f1(y, predicted, z.shape[1])
        results.append({"strength": float(strength), "accuracy": accuracy, "macro_f1": f1,
                        "score": 0.5 * accuracy + 0.5 * f1})
    best = max(results, key=lambda item: (round(item["score"], 6), -item["strength"]))
    return best["strength"], results
