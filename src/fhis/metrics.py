from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if len(np.unique(y_true)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true, scores))
        metrics["auprc"] = float(average_precision_score(y_true, scores))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    return metrics


def trace_ranking_metrics(rows: list[dict[str, Any]], scores: np.ndarray) -> dict[str, float]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        grouped[str(row["trace_id"])].append((row, float(score)))

    wrong_traces = [
        pairs for pairs in grouped.values() if any(int(row["label"]) == 1 for row, _ in pairs)
    ]
    if not wrong_traces:
        return {
            "recall_at_1": float("nan"),
            "recall_at_2": float("nan"),
            "top_30pct_budget_coverage": float("nan"),
            "n_wrong_traces": 0.0,
        }

    r1 = 0
    r2 = 0
    budget_hits = 0
    for pairs in wrong_traces:
        ranked = sorted(pairs, key=lambda x: x[1], reverse=True)
        labels = [int(row["label"]) for row, _ in ranked]
        r1 += int(labels[0] == 1)
        r2 += int(any(label == 1 for label in labels[:2]))
        budget = max(1, int(math.ceil(0.30 * len(labels))))
        budget_hits += int(any(label == 1 for label in labels[:budget]))

    denom = len(wrong_traces)
    return {
        "recall_at_1": r1 / denom,
        "recall_at_2": r2 / denom,
        "top_30pct_budget_coverage": budget_hits / denom,
        "n_wrong_traces": float(denom),
    }


def evaluate_scores(
    rows: list[dict[str, Any]],
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    out = binary_metrics(y_true, scores)
    out.update(trace_ranking_metrics(rows, scores))
    return out
