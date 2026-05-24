from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from fhis.train_bad_step_hazard_v21 import (
    HazardProbeArtifact,
    HazardProbeNet,
    build_dataset,
    calibrator_predict,
    labels,
    predict_logits,
)

# Some remote artifacts were produced by running the training module as a script,
# so joblib recorded these classes under __main__. Provide aliases before load.
import __main__  # noqa: E402

setattr(__main__, "HazardProbeArtifact", HazardProbeArtifact)
setattr(__main__, "HazardProbeNet", HazardProbeNet)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_artifact(
    artifact_path: Path,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    payload = joblib.load(artifact_path)
    artifact = payload["model"]
    config = payload.get("config", {})
    cfg = config.get("bad_step_hazard_v21", {})
    model = artifact.build_model().to(device)
    dataset = build_dataset(
        rows,
        artifact.feature_mean,
        artifact.feature_scale,
        artifact.scalar_mean,
        artifact.scalar_scale,
        bool(cfg.get("use_scalars", True)),
        len(artifact.layer_ids),
    )
    logits = predict_logits(model, dataset, device, int(cfg.get("eval_trace_batch_size", 64)))
    raw = 1.0 / (1.0 + np.exp(-logits))
    method = str(payload.get("probe_kind", "")).rsplit("_", 1)[-1]
    if method not in {"platt", "isotonic"}:
        method = str(payload.get("calibration_method", cfg.get("calibration_method", "platt")))
    calibrated = calibrator_predict(payload["calibrator"], logits, method)
    return raw, calibrated


def threshold_metrics(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y = labels(rows)
    pred = scores >= threshold
    pos = y == 1
    neg = y == 0
    tp = int((pred & pos).sum())
    fp = int((pred & neg).sum())
    fn = int((~pred & pos).sum())
    tn = int((~pred & neg).sum())

    grouped: dict[str, list[tuple[dict[str, Any], bool]]] = defaultdict(list)
    for row, flag in zip(rows, pred, strict=True):
        grouped[str(row["trace_id"])].append((row, bool(flag)))

    correct = 0
    correct_stopped = 0
    traces_with_fhis = 0
    traces_with_fhis_detected = 0
    prefhis_stopped = 0
    first_trigger_positive = 0
    first_trigger_total = 0
    for pairs in grouped.values():
        pairs = sorted(pairs, key=lambda x: int(x[0]["step_index"]))
        triggered = [(row, flag) for row, flag in pairs if flag]
        has_pos = any(int(row["label"]) == 1 for row, _ in pairs)
        is_correct = bool(pairs[0][0].get("trace_final_correct", False))
        if is_correct:
            correct += 1
            correct_stopped += int(bool(triggered))
        if has_pos:
            traces_with_fhis += 1
            fhis_step = min(int(row["step_index"]) for row, _ in pairs if int(row["label"]) == 1)
            traces_with_fhis_detected += int(any(flag and int(row["label"]) == 1 for row, flag in pairs))
            prefhis_stopped += int(any(flag and int(row["step_index"]) < fhis_step for row, flag in pairs))
        if triggered:
            first_trigger_total += 1
            first_trigger_positive += int(int(triggered[0][0]["label"]) == 1)

    first_trigger_fp = first_trigger_total - first_trigger_positive
    return {
        "threshold": float(threshold),
        "step_recall": tp / max(tp + fn, 1),
        "step_precision": tp / max(tp + fp, 1),
        "step_fp_per_tp": fp / max(tp, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "wrong_trace_recall": traces_with_fhis_detected / max(traces_with_fhis, 1),
        "correct_trace_false_stop": correct_stopped / max(correct, 1),
        "pre_fhis_false_stop": prefhis_stopped / max(traces_with_fhis, 1),
        "first_trigger_precision": first_trigger_positive / max(first_trigger_total, 1),
        "first_trigger_fp_per_tp": first_trigger_fp / max(first_trigger_positive, 1),
        "triggered_traces": first_trigger_total,
        "first_trigger_tp": first_trigger_positive,
        "first_trigger_fp": first_trigger_fp,
    }


def build_eval_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = labels(rows).astype(bool)
    trace_ids = [str(row["trace_id"]) for row in rows]
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, trace_id in enumerate(trace_ids):
        grouped[trace_id].append(idx)
    groups = []
    for indices in grouped.values():
        indices = sorted(indices, key=lambda i: int(rows[i]["step_index"]))
        group_y = y[indices]
        group_steps = np.asarray([int(rows[i]["step_index"]) for i in indices], dtype=np.int32)
        has_pos = bool(group_y.any())
        fhis_step = int(group_steps[group_y].min()) if has_pos else None
        groups.append(
            {
                "indices": np.asarray(indices, dtype=np.int32),
                "is_correct": bool(rows[indices[0]].get("trace_final_correct", False)),
                "has_pos": has_pos,
                "fhis_step": fhis_step,
                "step_indices": group_steps,
            }
        )
    return {
        "y": y,
        "groups": groups,
        "num_correct": sum(int(group["is_correct"]) for group in groups),
        "num_with_fhis": sum(int(group["has_pos"]) for group in groups),
    }


def threshold_metrics_fast(ctx: dict[str, Any], scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y = ctx["y"]
    pred = scores >= threshold
    pos = y
    neg = ~y
    tp = int((pred & pos).sum())
    fp = int((pred & neg).sum())
    fn = int((~pred & pos).sum())
    tn = int((~pred & neg).sum())

    correct_stopped = 0
    traces_with_fhis_detected = 0
    prefhis_stopped = 0
    first_trigger_positive = 0
    first_trigger_total = 0
    for group in ctx["groups"]:
        indices = group["indices"]
        group_pred = pred[indices]
        if not group_pred.any():
            continue
        first_pos = int(np.flatnonzero(group_pred)[0])
        first_trigger_total += 1
        if group["is_correct"]:
            correct_stopped += 1
        if group["has_pos"]:
            group_y = y[indices]
            traces_with_fhis_detected += int((group_pred & group_y).any())
            prefhis_stopped += int(
                bool((group_pred & (group["step_indices"] < int(group["fhis_step"]))).any())
            )
            first_trigger_positive += int(bool(group_y[first_pos]))

    first_trigger_fp = first_trigger_total - first_trigger_positive
    return {
        "threshold": float(threshold),
        "step_recall": tp / max(tp + fn, 1),
        "step_precision": tp / max(tp + fp, 1),
        "step_fp_per_tp": fp / max(tp, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "wrong_trace_recall": traces_with_fhis_detected / max(int(ctx["num_with_fhis"]), 1),
        "correct_trace_false_stop": correct_stopped / max(int(ctx["num_correct"]), 1),
        "pre_fhis_false_stop": prefhis_stopped / max(int(ctx["num_with_fhis"]), 1),
        "first_trigger_precision": first_trigger_positive / max(first_trigger_total, 1),
        "first_trigger_fp_per_tp": first_trigger_fp / max(first_trigger_positive, 1),
        "triggered_traces": first_trigger_total,
        "first_trigger_tp": first_trigger_positive,
        "first_trigger_fp": first_trigger_fp,
    }


def best_under_caps(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    caps: list[float],
    require_first_trigger_budget: float | None,
) -> list[dict[str, Any]]:
    ctx = build_eval_context(rows)
    return best_under_caps_with_context(ctx, scores, caps, require_first_trigger_budget)


def best_under_caps_with_context(
    ctx: dict[str, Any],
    scores: np.ndarray,
    caps: list[float],
    require_first_trigger_budget: float | None,
) -> list[dict[str, Any]]:
    return choose_from_threshold_table(
        score_threshold_table(ctx, scores),
        caps,
        require_first_trigger_budget,
    )


def score_threshold_table(ctx: dict[str, Any], scores: np.ndarray) -> list[dict[str, Any]]:
    candidates = np.unique(scores)
    return [threshold_metrics_fast(ctx, scores, float(tau)) for tau in candidates]


def choose_from_threshold_table(
    table_all: list[dict[str, Any]],
    caps: list[float],
    require_first_trigger_budget: float | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cap in caps:
        feasible: list[dict[str, Any]] = []
        for item in table_all:
            if item["correct_trace_false_stop"] >= cap:
                continue
            if require_first_trigger_budget is not None and item["first_trigger_fp_per_tp"] > require_first_trigger_budget:
                continue
            feasible.append(item)
        if feasible:
            best = max(
                feasible,
                key=lambda x: (
                    x["step_recall"],
                    x["wrong_trace_recall"],
                    -x["correct_trace_false_stop"],
                    -x["first_trigger_fp_per_tp"],
                    x["threshold"],
                ),
            )
        else:
            best = min(table_all, key=lambda x: (-x["threshold"], x["correct_trace_false_stop"]))
        item = dict(best)
        item["cap"] = float(cap)
        item["first_trigger_budget"] = require_first_trigger_budget
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score v2.5 hazard classifiers on clean eval.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, default=Path("classifier/v2_runs/bad_step_v25_train2000"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--caps", default="0.5,0.4,0.3,0.2")
    parser.add_argument("--first-trigger-budget", type=float, default=2.0)
    args = parser.parse_args()

    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    rows = payload["rows"]
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    model_dirs = {
        "base_platt": "hazard_recall_base_platt",
        "big_platt": "hazard_recall_big_platt",
        "trade_neg3_pos5": "hazard_tradeoff_base_neg3_pos5_platt",
        "trade_neg4_pos4": "hazard_tradeoff_base_neg4_pos4_platt",
    }
    scores: dict[str, np.ndarray] = {}
    raw_scores: dict[str, np.ndarray] = {}
    for name, dirname in model_dirs.items():
        raw, calibrated = score_artifact(args.models_root / dirname / "probe_calibrated.joblib", rows, device)
        raw_scores[f"{name}_raw"] = raw
        scores[name] = calibrated

    scores["avg_big_base"] = (scores["big_platt"] + scores["base_platt"]) / 2.0
    scores["avg_all4"] = (scores["big_platt"] + scores["base_platt"] + scores["trade_neg3_pos5"] + scores["trade_neg4_pos4"]) / 4.0
    scores["max_all4"] = np.maximum.reduce([scores["big_platt"], scores["base_platt"], scores["trade_neg3_pos5"], scores["trade_neg4_pos4"]])
    scores["min_all4"] = np.minimum.reduce([scores["big_platt"], scores["base_platt"], scores["trade_neg3_pos5"], scores["trade_neg4_pos4"]])

    y = labels(rows)
    caps = [float(x) for x in args.caps.split(",") if x.strip()]
    results: dict[str, Any] = {
        "features": str(args.features),
        "rows": len(rows),
        "traces": len({str(row["trace_id"]) for row in rows}),
        "correct_traces": len({str(row["trace_id"]) for row in rows if bool(row.get("trace_final_correct", False))}),
        "traces_with_fhis": len({str(row["trace_id"]) for row in rows if int(row.get("label", 0)) == 1}),
        "step_positives": int(y.sum()),
        "step_negatives": int(len(y) - y.sum()),
        "caps": caps,
        "budgeted_first_trigger_fp_per_tp": args.first_trigger_budget,
        "score_results": {},
        "best_by_cap_budgeted": [],
        "best_by_cap_unbudgeted": [],
    }
    ctx = build_eval_context(rows)
    tables = {name: score_threshold_table(ctx, score) for name, score in scores.items()}

    for name in scores:
        results["score_results"][name] = {
            "under_correct_false_stop_caps_budgeted": choose_from_threshold_table(
                tables[name], caps, args.first_trigger_budget
            ),
            "under_correct_false_stop_caps_unbudgeted": choose_from_threshold_table(
                tables[name], caps, None
            ),
        }

    for cap in caps:
        budgeted = []
        unbudgeted = []
        for name in scores:
            b = choose_from_threshold_table(tables[name], [cap], args.first_trigger_budget)[0]
            u = choose_from_threshold_table(tables[name], [cap], None)[0]
            b["score"] = name
            u["score"] = name
            budgeted.append(b)
            unbudgeted.append(u)
        results["best_by_cap_budgeted"].append(
            max(
                budgeted,
                key=lambda x: (
                    x["step_recall"],
                    x["wrong_trace_recall"],
                    -x["correct_trace_false_stop"],
                    -x["first_trigger_fp_per_tp"],
                ),
            )
        )
        results["best_by_cap_unbudgeted"].append(
            max(
                unbudgeted,
                key=lambda x: (
                    x["step_recall"],
                    x["wrong_trace_recall"],
                    -x["correct_trace_false_stop"],
                    -x["first_trigger_fp_per_tp"],
                ),
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
