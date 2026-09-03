"""用途：提供二分类概率的阈值候选、指标计算与验证集阈值选择。"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _validated_arrays(
    labels: Iterable[int], positive_probabilities: Iterable[float]
) -> Tuple[np.ndarray, np.ndarray]:
    label_array = np.asarray(tuple(labels), dtype=np.int64)
    probability_array = np.asarray(tuple(positive_probabilities), dtype=np.float64)
    if label_array.ndim != 1 or probability_array.ndim != 1:
        raise ValueError("labels and positive_probabilities must be one-dimensional.")
    if label_array.size == 0 or label_array.size != probability_array.size:
        raise ValueError("labels and positive_probabilities must be non-empty and aligned.")
    if not np.isin(label_array, (0, 1)).all():
        raise ValueError("labels must contain only binary values 0 and 1.")
    if not np.isfinite(probability_array).all():
        raise ValueError("positive_probabilities must be finite.")
    if ((probability_array < 0.0) | (probability_array > 1.0)).any():
        raise ValueError("positive_probabilities must be in [0, 1].")
    return label_array, probability_array


def threshold_candidates(positive_probabilities: Iterable[float]) -> Tuple[float, ...]:
    """返回覆盖所有不同预测划分的中点阈值，并显式包含 0.5。"""
    probabilities = np.asarray(tuple(positive_probabilities), dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("positive_probabilities must be a non-empty one-dimensional sequence.")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0.0) | (probabilities > 1.0)
    ).any():
        raise ValueError("positive_probabilities must be finite and in [0, 1].")
    unique = np.unique(probabilities)
    candidates = {0.0, 0.5, 1.0}
    candidates.update(float((left + right) / 2.0) for left, right in zip(unique[:-1], unique[1:]))
    return tuple(sorted(candidates))


def binary_metrics_at_threshold(
    labels: Iterable[int],
    positive_probabilities: Iterable[float],
    threshold: float,
) -> Dict[str, float]:
    """按给定阈值计算二分类指标和混淆矩阵。"""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    label_array, probability_array = _validated_arrays(labels, positive_probabilities)
    predictions = (probability_array >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(label_array, predictions, labels=[0, 1]).ravel()
    metrics = {
        "threshold": float(threshold),
        "accuracy": accuracy_score(label_array, predictions),
        "macro_f1": f1_score(label_array, predictions, average="macro", zero_division=0),
        "positive_f1": f1_score(label_array, predictions, pos_label=1, zero_division=0),
        "precision": precision_score(label_array, predictions, pos_label=1, zero_division=0),
        "recall": recall_score(label_array, predictions, pos_label=1, zero_division=0),
        "auroc": (
            roc_auc_score(label_array, probability_array)
            if np.unique(label_array).size > 1
            else float("nan")
        ),
        "negative_predictions": int((predictions == 0).sum()),
        "positive_predictions": int((predictions == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return {key: float(value) for key, value in metrics.items()}


def select_validation_threshold(
    labels: Iterable[int], positive_probabilities: Iterable[float]
) -> Dict[str, object]:
    """仅按验证集选择阈值，并使用确定性的多级并列规则。"""
    label_array, probability_array = _validated_arrays(labels, positive_probabilities)
    candidates = threshold_candidates(probability_array)
    evaluated = [
        binary_metrics_at_threshold(label_array, probability_array, threshold)
        for threshold in candidates
    ]
    best = max(
        evaluated,
        key=lambda item: (
            item["positive_f1"],
            item["macro_f1"],
            -abs(item["threshold"] - 0.5),
            item["threshold"],
        ),
    )
    return {
        "selection_metric": "positive_f1",
        "tie_break": ["macro_f1", "closest_to_0.5", "higher_threshold"],
        "candidate_count": len(candidates),
        "threshold": best["threshold"],
        "metrics": best,
    }
