from __future__ import annotations

import numpy as np

from fhis.online_eval import (
    fhis_recall_constrained_metrics,
    online_metrics,
    threshold_for_target_fhis_step_recall,
    threshold_for_target_recall,
)


def row(trace_id: str, step_index: int, label: int) -> dict:
    return {
        "trace_id": trace_id,
        "problem_id": trace_id,
        "step_index": step_index,
        "label": label,
        "feature": None,
        "baselines": {},
    }


def test_online_metrics_counts_caught_missed_and_false_stops() -> None:
    rows = [
        row("wrong-caught", 1, 0),
        row("wrong-caught", 2, 1),
        row("wrong-missed", 1, 0),
        row("wrong-missed", 2, 1),
        row("correct-stop", 1, 0),
        row("correct-stop", 2, 0),
        row("correct-clean", 1, 0),
        row("correct-clean", 2, 0),
    ]
    scores = np.asarray([0.1, 0.8, 0.1, 0.2, 0.7, 0.1, 0.1, 0.2])

    metrics = online_metrics(rows, scores, threshold=0.5)

    assert metrics["online_recall_by_fhis"] == 0.5
    assert metrics["miss_rate_by_fhis"] == 0.5
    assert metrics["correct_false_stop_rate"] == 0.5
    assert metrics["alert_at_fhis_rate"] == 0.5


def test_threshold_for_target_recall_uses_highest_specific_threshold() -> None:
    rows = [
        row("wrong-1", 1, 0),
        row("wrong-1", 2, 1),
        row("wrong-2", 1, 1),
        row("correct", 1, 0),
    ]
    scores = np.asarray([0.1, 0.8, 0.6, 0.7])

    threshold = threshold_for_target_recall(rows, scores, target_recall=1.0)

    assert threshold == 0.6


def test_fhis_recall_constrained_metrics_excludes_post_fhis_steps() -> None:
    rows = [
        row("wrong", 1, 0),
        row("wrong", 2, 1),
        row("wrong", 3, 0),
        row("correct", 1, 0),
        row("correct", 2, 0),
    ]
    scores = np.asarray([0.6, 0.8, 0.9, 0.7, 0.1])

    metrics = fhis_recall_constrained_metrics(rows, scores, threshold=0.75)

    assert metrics["fhis_step_recall"] == 1.0
    assert metrics["observable_non_fhis_steps"] == 3.0
    assert metrics["observable_non_fhis_step_fpr"] == 0.0
    assert metrics["correct_trace_false_stop_rate"] == 0.0


def test_threshold_for_target_fhis_step_recall_reports_step_operating_point() -> None:
    rows = [
        row("wrong-1", 1, 0),
        row("wrong-1", 2, 1),
        row("wrong-2", 1, 1),
        row("correct", 1, 0),
    ]
    scores = np.asarray([0.1, 0.8, 0.6, 0.7])

    threshold = threshold_for_target_fhis_step_recall(rows, scores, target_recall=1.0)

    assert threshold == 0.6
