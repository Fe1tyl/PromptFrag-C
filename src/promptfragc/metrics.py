"""Numpy implementations of the frozen evaluation metrics."""

from __future__ import annotations

import numpy as np


def _validate(probabilities: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(targets, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError(f"probabilities must have shape [N, C], got {probs.shape}")
    if labels.ndim != 1 or labels.shape[0] != probs.shape[0]:
        raise ValueError("targets must have shape [N] and match probabilities")
    if probs.shape[0] == 0:
        raise ValueError("metrics require at least one example")
    if not np.isfinite(probs).all():
        raise ValueError("probabilities contain non-finite values")
    if np.any(probs < -1e-7) or np.any(probs > 1.0 + 1e-7):
        raise ValueError("probabilities must lie in [0, 1]")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError("probability rows must sum to one")
    if np.any(labels < 0) or np.any(labels >= probs.shape[1]):
        raise ValueError("target index outside class range")
    return probs, labels


def expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Equal-width ECE over maximum-class confidence."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    conf = np.asarray(confidence, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    if conf.shape != corr.shape or conf.ndim != 1:
        raise ValueError("confidence and correct must be matching vectors")
    bin_ids = np.minimum((conf * n_bins).astype(np.int64), n_bins - 1)
    ece = 0.0
    total = conf.size
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        weight = float(mask.sum()) / float(total)
        ece += weight * abs(float(corr[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def area_under_risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
    """Return AURC and risk at 80% coverage.

    Examples are accepted from highest to lowest confidence. AURC is the mean
    cumulative error rate across all attainable coverage levels.
    """
    conf = np.asarray(confidence, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    if conf.shape != corr.shape or conf.ndim != 1 or conf.size == 0:
        raise ValueError("confidence and correct must be non-empty matching vectors")
    order = np.argsort(-conf, kind="stable")
    errors = 1.0 - corr[order]
    cumulative_risk = np.cumsum(errors) / np.arange(1, errors.size + 1)
    cutoff = max(1, int(np.ceil(0.8 * errors.size)))
    return float(cumulative_risk.mean()), float(cumulative_risk[cutoff - 1])


def classification_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> dict[str, float]:
    """Compute all preregistered metrics for one prediction matrix."""
    probs, labels = _validate(probabilities, targets)
    predictions = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = predictions == labels
    true_probability = probs[np.arange(labels.size), labels]
    nll = -np.log(np.clip(true_probability, 1e-12, 1.0)).mean()

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(labels.size), labels] = 1.0
    brier = np.square(probs - one_hot).sum(axis=1).mean()
    aurc, risk_at_80 = area_under_risk_coverage(confidence, correct)

    result = {
        "accuracy": float(correct.mean()),
        "ece": expected_calibration_error(confidence, correct, n_bins=n_bins),
        "nll": float(nll),
        "brier": float(brier),
        "aurc": aurc,
        "risk_at_80": risk_at_80,
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError("metric computation produced a non-finite value")
    return result

