from __future__ import annotations

from typing import Any

import numpy as np


def l2_normalize(features: np.ndarray) -> np.ndarray:
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def pair_scores(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = l2_normalize(features.astype(np.float32))
    scores: list[np.ndarray] = []
    same: list[np.ndarray] = []
    for i in range(len(features) - 1):
        scores.append((features[i + 1 :] @ features[i]).astype(np.float32))
        same.append(labels[i + 1 :] == labels[i])
    if not scores:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.bool_)
    return np.concatenate(scores), np.concatenate(same)


def roc_arrays(scores: np.ndarray, same: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positives = float(same.sum())
    negatives = float((~same).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Need both genuine and impostor pairs.")
    order = np.argsort(-scores, kind="mergesort")
    sorted_same = same[order]
    sorted_scores = scores[order]
    tp = np.cumsum(sorted_same, dtype=np.float64)
    fp = np.cumsum(~sorted_same, dtype=np.float64)
    return fp / negatives, tp / positives, sorted_scores


def compute_metrics(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(str)
    scores, same = pair_scores(features, labels)
    false_accept, true_accept, thresholds = roc_arrays(scores, same)
    miss = 1.0 - true_accept
    idx = int(np.nanargmin(np.abs(miss - false_accept)))
    threshold = float(thresholds[idx])
    predictions = scores >= threshold
    accuracy = float(np.mean(predictions == same))
    eer = float((false_accept[idx] + miss[idx]) / 2.0)
    return {
        "eer": eer,
        "eer_percent": eer * 100.0,
        "eer_threshold": threshold,
        "verification_accuracy": accuracy,
        "verification_accuracy_percent": accuracy * 100.0,
        "genuine_pairs": int(same.sum()),
        "impostor_pairs": int((~same).sum()),
        "num_samples": int(len(labels)),
        "num_identities": int(len(set(labels))),
    }
