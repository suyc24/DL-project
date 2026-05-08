from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from fhis.train_probe import (
    HiddenStateMLP,
    RecallBiasedLoss,
    TorchMLPProbe,
    stack_features,
)


def grouped_by_trace(rows: list[dict[str, Any]], scores: np.ndarray) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        item = dict(row)
        item["score"] = float(score)
        grouped[str(row["trace_id"])].append(item)
    for trace_rows in grouped.values():
        trace_rows.sort(key=lambda row: int(row["step_index"]))
    return grouped


def trace_outcome(trace_rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    labels = [int(row["label"]) for row in trace_rows]
    scores = [float(row["score"]) for row in trace_rows]
    step_indices = [int(row["step_index"]) for row in trace_rows]
    is_wrong = any(label == 1 for label in labels)
    alert_positions = [idx for idx, score in enumerate(scores) if score >= threshold]
    first_alert_pos = alert_positions[0] if alert_positions else None
    first_alert_step = step_indices[first_alert_pos] if first_alert_pos is not None else None

    if is_wrong:
        fhis_pos = labels.index(1)
        fhis_step = step_indices[fhis_pos]
        caught = first_alert_pos is not None and first_alert_pos <= fhis_pos
        return {
            "is_wrong": True,
            "caught_by_fhis": caught,
            "missed_by_fhis": not caught,
            "pre_fhis_stop": first_alert_pos is not None and first_alert_pos < fhis_pos,
            "alert_at_fhis": scores[fhis_pos] >= threshold,
            "first_alert_step": first_alert_step,
            "fhis_step": fhis_step,
            "steps_until_stop": first_alert_step if caught else fhis_step,
            "observed_steps": fhis_step,
        }

    observed_steps = max(step_indices) if step_indices else 0
    false_stop = first_alert_pos is not None
    return {
        "is_wrong": False,
        "correct_false_stop": false_stop,
        "first_alert_step": first_alert_step,
        "steps_until_stop": first_alert_step if false_stop else observed_steps,
        "observed_steps": observed_steps,
    }


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def online_metrics(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> dict[str, float]:
    grouped = grouped_by_trace(rows, scores)
    outcomes = [trace_outcome(trace_rows, threshold) for trace_rows in grouped.values()]
    wrong = [item for item in outcomes if item["is_wrong"]]
    correct = [item for item in outcomes if not item["is_wrong"]]
    caught = [item for item in wrong if item["caught_by_fhis"]]
    missed = [item for item in wrong if item["missed_by_fhis"]]
    correct_false = [item for item in correct if item["correct_false_stop"]]

    total_observed_steps = sum(float(item["observed_steps"]) for item in outcomes)
    total_steps_until_stop = sum(float(item["steps_until_stop"]) for item in outcomes)
    return {
        "threshold": float(threshold),
        "traces": float(len(outcomes)),
        "wrong_traces": float(len(wrong)),
        "correct_traces": float(len(correct)),
        "online_recall_by_fhis": len(caught) / len(wrong) if wrong else float("nan"),
        "miss_rate_by_fhis": len(missed) / len(wrong) if wrong else float("nan"),
        "pre_fhis_stop_rate_on_wrong": (
            sum(bool(item["pre_fhis_stop"]) for item in wrong) / len(wrong)
            if wrong
            else float("nan")
        ),
        "alert_at_fhis_rate": (
            sum(bool(item["alert_at_fhis"]) for item in wrong) / len(wrong)
            if wrong
            else float("nan")
        ),
        "correct_false_stop_rate": len(correct_false) / len(correct) if correct else float("nan"),
        "avg_stop_step_wrong": mean_or_nan([float(item["steps_until_stop"]) for item in wrong]),
        "avg_fhis_step": mean_or_nan([float(item["fhis_step"]) for item in wrong]),
        "avg_stop_step_correct": mean_or_nan(
            [float(item["steps_until_stop"]) for item in correct]
        ),
        "avg_observed_steps_per_trace": total_observed_steps / len(outcomes) if outcomes else 0.0,
        "avg_steps_until_stop_per_trace": total_steps_until_stop / len(outcomes)
        if outcomes
        else 0.0,
        "step_compute_fraction": total_steps_until_stop / total_observed_steps
        if total_observed_steps
        else float("nan"),
    }


def fhis_recall_constrained_metrics(
    rows: list[dict[str, Any]], scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Evaluate false positives when the actual FHIS step must be detected.

    Negative steps are the steps an online policy can observe before the FHIS:
    all steps from correct traces and only pre-FHIS steps from wrong traces.
    Post-FHIS steps are excluded because online inference should already have
    stopped or branched at the first harmful step.
    """
    grouped = grouped_by_trace(rows, scores)
    fhis_scores: list[float] = []
    negative_scores: list[float] = []
    correct_trace_alerts = 0
    correct_traces = 0
    wrong_pre_fhis_alerts = 0
    wrong_traces = 0

    for trace_rows in grouped.values():
        labels = [int(row["label"]) for row in trace_rows]
        trace_scores = [float(row["score"]) for row in trace_rows]
        if any(label == 1 for label in labels):
            wrong_traces += 1
            fhis_pos = labels.index(1)
            fhis_scores.append(trace_scores[fhis_pos])
            pre_scores = trace_scores[:fhis_pos]
            negative_scores.extend(pre_scores)
            wrong_pre_fhis_alerts += int(any(score >= threshold for score in pre_scores))
        else:
            correct_traces += 1
            negative_scores.extend(trace_scores)
            correct_trace_alerts += int(any(score >= threshold for score in trace_scores))

    fhis_arr = np.asarray(fhis_scores, dtype=np.float64)
    neg_arr = np.asarray(negative_scores, dtype=np.float64)
    false_positive_steps = int((neg_arr >= threshold).sum()) if len(neg_arr) else 0
    return {
        "threshold": float(threshold),
        "fhis_steps": float(len(fhis_arr)),
        "observable_non_fhis_steps": float(len(neg_arr)),
        "fhis_step_recall": float((fhis_arr >= threshold).mean()) if len(fhis_arr) else float("nan"),
        "observable_non_fhis_step_fpr": float((neg_arr >= threshold).mean())
        if len(neg_arr)
        else float("nan"),
        "false_positive_steps": float(false_positive_steps),
        "correct_trace_false_stop_rate": correct_trace_alerts / correct_traces
        if correct_traces
        else float("nan"),
        "pre_fhis_stop_rate_on_wrong": wrong_pre_fhis_alerts / wrong_traces
        if wrong_traces
        else float("nan"),
    }


def threshold_for_target_fhis_step_recall(
    rows: list[dict[str, Any]], scores: np.ndarray, target_recall: float
) -> float:
    candidates = np.unique(np.asarray(scores, dtype=np.float64))
    candidates = np.concatenate(
        [
            [float(np.nextafter(candidates.min(), -np.inf))],
            candidates,
            [float(np.nextafter(candidates.max(), np.inf))],
        ]
    )
    best_threshold = float(candidates.min())
    best_fpr = float("inf")
    for threshold in candidates:
        metrics = fhis_recall_constrained_metrics(rows, scores, float(threshold))
        recall = metrics["fhis_step_recall"]
        fpr = metrics["observable_non_fhis_step_fpr"]
        # Higher threshold wins ties because it is the stricter operating point.
        if recall >= target_recall and (fpr < best_fpr or fpr == best_fpr):
            best_threshold = float(threshold)
            best_fpr = float(fpr)
    return best_threshold


def threshold_for_target_recall(
    rows: list[dict[str, Any]], scores: np.ndarray, target_recall: float
) -> float:
    candidates = np.unique(np.asarray(scores, dtype=np.float64))
    candidates = np.concatenate(
        [
            [float(np.nextafter(candidates.min(), -np.inf))],
            candidates,
            [float(np.nextafter(candidates.max(), np.inf))],
        ]
    )
    best_threshold = float(candidates.min())
    best_false_stop = float("inf")
    for threshold in candidates:
        metrics = online_metrics(rows, scores, float(threshold))
        recall = metrics["online_recall_by_fhis"]
        false_stop = metrics["correct_false_stop_rate"]
        if recall >= target_recall and false_stop <= best_false_stop:
            best_threshold = float(threshold)
            best_false_stop = float(false_stop)
    return best_threshold


def fixed_thresholds(default_threshold: float) -> list[float]:
    values = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, default_threshold]
    return sorted({float(value) for value in values if 0.0 <= float(value) <= 1.0})


def load_probe(path: str | Path) -> dict[str, Any]:
    # Older MLP artifacts were saved from a `python -m` entrypoint and refer to __main__.
    import __main__

    __main__.TorchMLPProbe = TorchMLPProbe
    __main__.HiddenStateMLP = HiddenStateMLP
    __main__.RecallBiasedLoss = RecallBiasedLoss
    return joblib.load(path)


def score_model(probe_payload: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    model = probe_payload["model"]
    return model.predict_proba(stack_features(rows))[:, 1]


def evaluate_model(
    name: str,
    probe_payload: dict[str, Any],
    rows: list[dict[str, Any]],
    target_recalls: list[float],
) -> dict[str, Any]:
    scores = score_model(probe_payload, rows)
    model = probe_payload["model"]
    default_threshold = float(getattr(model, "decision_threshold", 0.5))
    fixed = {
        str(threshold): online_metrics(rows, scores, threshold)
        for threshold in fixed_thresholds(default_threshold)
    }
    oracle = {}
    fhis_step_oracle = {}
    for target in target_recalls:
        threshold = threshold_for_target_recall(rows, scores, target)
        oracle[str(target)] = online_metrics(rows, scores, threshold)
        fhis_step_threshold = threshold_for_target_fhis_step_recall(rows, scores, target)
        fhis_step_oracle[str(target)] = fhis_recall_constrained_metrics(
            rows, scores, fhis_step_threshold
        )
    return {
        "name": name,
        "default_threshold": default_threshold,
        "fixed_thresholds": fixed,
        "oracle_target_recall_thresholds": oracle,
        "fhis_step_recall_constrained_thresholds": fhis_step_oracle,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate probes as online stop/continue policies.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--mlp-probe", required=True)
    parser.add_argument("--logistic-probe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-recall", type=float, action="append", default=[0.95, 0.99, 1.0])
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    rows = payload["rows"]
    results = {
        "features": args.features,
        "rows": len(rows),
        "traces": len({row["trace_id"] for row in rows}),
        "models": {
            "hidden_mlp": evaluate_model(
                "hidden_mlp", load_probe(args.mlp_probe), rows, args.target_recall
            ),
            "hidden_logistic": evaluate_model(
                "hidden_logistic", load_probe(args.logistic_probe), rows, args.target_recall
            ),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
