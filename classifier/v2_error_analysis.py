from __future__ import annotations

import argparse
import json
import math
import sys
import __main__
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fhis.io import read_jsonl  # noqa: E402
from fhis.metrics import evaluate_scores  # noqa: E402
from fhis.online_router import load_probe  # noqa: E402
from fhis.train_probe import labels, rows_for_split, split_problem_ids, stack_features  # noqa: E402
from fhis import train_probe_v2  # noqa: E402


for _name in ("LayerwiseProbeNet", "LayerwiseProbeArtifact"):
    if not hasattr(__main__, _name):
        setattr(__main__, _name, getattr(train_probe_v2, _name))


def score_with_probe(probe_artifact: str | Path, rows: list[dict[str, Any]]) -> np.ndarray:
    try:
        loaded = load_probe(probe_artifact)
    except Exception:
        loaded = joblib.load(probe_artifact)
    model = loaded.get("model", loaded) if isinstance(loaded, dict) else loaded
    x = stack_features(rows)
    if hasattr(model, "predict_scores"):
        try:
            return model.predict_scores(x, rows=rows)
        except TypeError:
            return model.predict_scores(x)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    raise TypeError(f"Probe {probe_artifact} does not expose predict_proba/predict_scores")


def finite(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def trace_groups(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
) -> dict[str, list[tuple[dict[str, Any], float]]]:
    groups: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        groups[str(row["trace_id"])].append((row, float(score)))
    for pairs in groups.values():
        pairs.sort(key=lambda item: int(item[0]["step_index"]))
    return groups


def threshold_for_recall(y_true: np.ndarray, scores: np.ndarray, target: float) -> float:
    positives = scores[y_true == 1]
    if len(positives) == 0:
        return float("inf")
    candidates = np.unique(positives)
    best_tau = float(candidates.min())
    best_fpr = float("inf")
    negatives = scores[y_true == 0]
    for tau in candidates:
        recall = float((positives >= tau).mean())
        if recall + 1e-12 < target:
            continue
        fpr = float((negatives >= tau).mean()) if len(negatives) else 0.0
        if fpr < best_fpr:
            best_fpr = fpr
            best_tau = float(tau)
    return best_tau


def threshold_for_budget(scores: np.ndarray, target_rate: float) -> float:
    if len(scores) == 0:
        return float("inf")
    target_rate = min(max(float(target_rate), 0.0), 1.0)
    if target_rate <= 0.0:
        return float(np.nextafter(scores.max(), np.inf))
    if target_rate >= 1.0:
        return float(np.nextafter(scores.min(), -np.inf))
    return float(np.quantile(scores, 1.0 - target_rate, method="lower"))


def online_prefix_metrics(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = labels(rows)
    positives = y_true == 1
    negatives = y_true == 0
    pred = scores >= threshold
    groups = trace_groups(rows, scores)

    wrong_groups = [
        pairs for pairs in groups.values() if any(int(row["label"]) == 1 for row, _ in pairs)
    ]
    correct_groups = [
        pairs
        for pairs in groups.values()
        if pairs and bool(pairs[0][0].get("trace_final_correct", False))
    ]
    detected_wrong = 0
    prefhis_stops = 0
    correct_false_stops = 0
    stop_steps_wrong: list[int] = []
    fhis_steps: list[int] = []
    stop_steps_correct: list[int] = []
    observed_steps = 0
    consumed_steps = 0

    for pairs in wrong_groups:
        positive_step = next(int(row["step_index"]) for row, _ in pairs if int(row["label"]) == 1)
        fhis_steps.append(positive_step)
        first_alert = next(
            (int(row["step_index"]) for row, score in pairs if score >= threshold),
            None,
        )
        observed_steps += len(pairs)
        if first_alert is None:
            consumed_steps += len(pairs)
            continue
        consumed_steps += max(1, sum(1 for row, _ in pairs if int(row["step_index"]) <= first_alert))
        stop_steps_wrong.append(first_alert)
        if first_alert <= positive_step:
            detected_wrong += 1
        if first_alert < positive_step:
            prefhis_stops += 1

    for pairs in correct_groups:
        first_alert = next(
            (int(row["step_index"]) for row, score in pairs if score >= threshold),
            None,
        )
        observed_steps += len(pairs)
        if first_alert is None:
            consumed_steps += len(pairs)
            continue
        correct_false_stops += 1
        consumed_steps += max(1, sum(1 for row, _ in pairs if int(row["step_index"]) <= first_alert))
        stop_steps_correct.append(first_alert)

    return {
        "threshold": float(threshold),
        "rows": float(len(rows)),
        "traces": float(len(groups)),
        "fhis_steps": float(positives.sum()),
        "non_fhis_steps": float(negatives.sum()),
        "flagged_steps": float(pred.sum()),
        "fhis_step_recall": float((pred & positives).sum() / max(positives.sum(), 1)),
        "observable_non_fhis_step_fpr": float((pred & negatives).sum() / max(negatives.sum(), 1)),
        "precision_step": float((pred & positives).sum() / max(pred.sum(), 1)),
        "wrong_traces": float(len(wrong_groups)),
        "correct_traces": float(len(correct_groups)),
        "online_recall_by_fhis": float(detected_wrong / max(len(wrong_groups), 1)),
        "pre_fhis_stop_rate_on_wrong": float(prefhis_stops / max(len(wrong_groups), 1)),
        "correct_trace_false_stop_rate": float(correct_false_stops / max(len(correct_groups), 1)),
        "avg_fhis_step": float(np.mean(fhis_steps)) if fhis_steps else float("nan"),
        "avg_stop_step_wrong": float(np.mean(stop_steps_wrong)) if stop_steps_wrong else float("nan"),
        "avg_stop_step_correct": float(np.mean(stop_steps_correct)) if stop_steps_correct else float("nan"),
        "step_compute_fraction": float(consumed_steps / max(observed_steps, 1)),
    }


def load_trace_map(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists() or p.stat().st_size < 200:
        return {}
    return {str(row["trace_id"]): row for row in read_jsonl(p)}


def candidate_row(
    row: dict[str, Any],
    score: float,
    traces: dict[str, dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    trace = traces.get(str(row["trace_id"]), {})
    return {
        "reason": reason,
        "trace_id": row["trace_id"],
        "problem_id": row["problem_id"],
        "subject": row.get("subject"),
        "level": row.get("level"),
        "step_index": int(row["step_index"]),
        "label": int(row["label"]),
        "trace_final_correct": bool(row.get("trace_final_correct", False)),
        "score": float(score),
        "step_length_chars": finite((row.get("baselines") or {}).get("step_length_chars")),
        "mean_token_logprob": finite((row.get("baselines") or {}).get("mean_token_logprob")),
        "problem": trace.get("problem"),
        "reference_answer": trace.get("reference_answer"),
        "final_answer": trace.get("final_answer"),
        "step_text": row.get("step_text", ""),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def conflict_rows(labels_path: str | Path | None, traces: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if labels_path is None:
        return []
    p = Path(labels_path)
    if not p.exists() or p.stat().st_size < 200:
        return []
    rows = []
    for label in read_jsonl(p):
        rough = label.get("rough_final_correct")
        final = label.get("final_correct")
        if rough is None or bool(rough) == bool(final):
            continue
        trace = traces.get(str(label["trace_id"]), {})
        rows.append(
            {
                "trace_id": label["trace_id"],
                "problem_id": label.get("problem_id"),
                "rough_final_correct": rough,
                "codex_final_correct": final,
                "first_invalid_step": label.get("first_invalid_step"),
                "confidence": label.get("confidence"),
                "reason": label.get("reason"),
                "problem": trace.get("problem"),
                "reference_answer": trace.get("reference_answer"),
                "final_answer": trace.get("final_answer"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="FHIS v2 probe error analysis and relabel queue.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--traces", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--output-dir", default="classifier/v2_runs/error_analysis")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260505)
    parser.add_argument("--top-k", type=int, default=300)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    rows = payload["rows"]
    if args.split != "all":
        splits = split_problem_ids(rows, args.train_frac, args.val_frac, args.seed)
        rows = rows_for_split(rows, splits[args.split])
    y_true = labels(rows)
    scores = score_with_probe(args.probe, rows)
    traces = load_trace_map(args.traces)

    thresholds: dict[str, float] = {"fixed_0_5": 0.5}
    for target in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
        thresholds[f"recall_{target:.2f}"] = threshold_for_recall(y_true, scores, target)
    for rate in [0.05, 0.10, 0.20, 0.30, 0.50]:
        thresholds[f"budget_{rate:.2f}"] = threshold_for_budget(scores, rate)

    summary = {
        "features": str(args.features),
        "probe": str(args.probe),
        "split": args.split,
        "rows": len(rows),
        "positive_steps": int(y_true.sum()),
        "negative_steps": int(len(y_true) - y_true.sum()),
        "offline": evaluate_scores(rows, y_true, scores),
        "thresholds": {
            name: online_prefix_metrics(rows, scores, tau) for name, tau in thresholds.items()
        },
    }

    scored = list(zip(rows, scores, strict=True))
    negatives = [(row, score) for row, score in scored if int(row["label"]) == 0]
    positives = [(row, score) for row, score in scored if int(row["label"]) == 1]
    hard_negatives = [
        candidate_row(row, score, traces, "high_score_non_fhis")
        for row, score in sorted(negatives, key=lambda item: item[1], reverse=True)[: args.top_k]
    ]
    false_negatives = [
        candidate_row(row, score, traces, "low_score_fhis")
        for row, score in sorted(positives, key=lambda item: item[1])[: args.top_k]
    ]
    correct_trace_fps = [
        candidate_row(row, score, traces, "high_score_correct_trace")
        for row, score in sorted(
            [(row, score) for row, score in negatives if bool(row.get("trace_final_correct", False))],
            key=lambda item: item[1],
            reverse=True,
        )[: args.top_k]
    ]
    conflicts = conflict_rows(args.labels, traces)

    out_dir = Path(args.output_dir)
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "hard_negative_candidates.jsonl", hard_negatives)
    write_jsonl(out_dir / "false_negative_candidates.jsonl", false_negatives)
    write_jsonl(out_dir / "correct_trace_false_positive_candidates.jsonl", correct_trace_fps)
    write_jsonl(out_dir / "rough_codex_conflicts.jsonl", conflicts)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote analysis to {out_dir}")


if __name__ == "__main__":
    main()
