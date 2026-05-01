from __future__ import annotations

import numpy as np

from fhis.metrics import evaluate_scores


def test_trace_ranking_metrics() -> None:
    rows = [
        {"trace_id": "a", "label": 0},
        {"trace_id": "a", "label": 1},
        {"trace_id": "b", "label": 0},
        {"trace_id": "b", "label": 1},
    ]
    scores = np.asarray([0.1, 0.9, 0.8, 0.2])
    y = np.asarray([0, 1, 0, 1])
    metrics = evaluate_scores(rows, y, scores)
    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_2"] == 1.0
