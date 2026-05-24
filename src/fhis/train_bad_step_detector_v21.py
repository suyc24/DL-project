from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from fhis.config import load_config
from fhis.metrics import evaluate_scores
from fhis.train_probe import labels, stack_features
from fhis.train_probe_v2 import train_layerwise_probe


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def problem_hash(problem_id: str) -> int:
    return int(hashlib.sha1(problem_id.encode()).hexdigest()[:8], 16) % 100


def stable_hash(value: str) -> int:
    return int(hashlib.sha1(value.encode()).hexdigest()[:8], 16) % 100


def inner_train_val(rows: list[dict[str, Any]], val_frac_pct: int = 15) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for row in rows:
        key = f"{row['problem_id']}::{row['trace_id']}"
        (val if stable_hash(key) < val_frac_pct else train).append(row)
    if not val or not train:
        return rows, rows
    return train, val


def rows_from_manifest(feature_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_by_key = {
        (str(obj["trace_id"]), int(obj["step_index"])): obj for obj in manifest_rows
    }
    out: list[dict[str, Any]] = []
    missing = 0
    for row in feature_rows:
        key = (str(row["trace_id"]), int(row["step_index"]))
        meta = manifest_by_key.get(key)
        if meta is None:
            missing += 1
            continue
        new_row = dict(row)
        new_row["label"] = int(meta["target_bad_step"])
        new_row["sample_weight"] = float(meta["sample_weight"])
        new_row["manifest_split"] = str(meta["split"])
        new_row["example_type"] = str(meta["example_type"])
        new_row["validity"] = str(meta["validity"])
        new_row["label_confidence"] = str(meta.get("label_confidence", "unknown"))
        new_row["first_invalid_step"] = meta.get("first_invalid_step")
        out.append(new_row)
    if missing:
        print(f"warning: skipped {missing} feature rows without manifest entries")
    return out


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "calibration": [], "hard_dev": []}
    for row in rows:
        split = str(row.get("manifest_split", "train"))
        splits.setdefault(split, []).append(row)
    return splits


def trace_stop_rates(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> dict[str, float]:
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        grouped[str(row["trace_id"])].append((row, float(score)))

    correct = 0
    correct_stopped = 0
    wrong_with_fhis = 0
    prefhis_stopped = 0
    first_trigger_positive = 0
    first_trigger_total = 0
    for pairs in grouped.values():
        pairs = sorted(pairs, key=lambda x: int(x[0]["step_index"]))
        triggered = [(row, score) for row, score in pairs if score >= threshold]
        has_pos = any(int(row["label"]) == 1 for row, _ in pairs)
        is_correct = bool(pairs[0][0].get("trace_final_correct", False))
        if is_correct:
            correct += 1
            correct_stopped += int(bool(triggered))
        if has_pos:
            wrong_with_fhis += 1
            fhis_step = min(int(row["step_index"]) for row, _ in pairs if int(row["label"]) == 1)
            prefhis_stopped += int(any(int(row["step_index"]) < fhis_step for row, _ in triggered))
        if triggered:
            first_trigger_total += 1
            first_row, _ = triggered[0]
            first_trigger_positive += int(int(first_row["label"]) == 1)

    return {
        "correct_trace_false_stop_rate": correct_stopped / max(correct, 1),
        "pre_fhis_false_stop_rate": prefhis_stopped / max(wrong_with_fhis, 1),
        "first_trigger_precision": first_trigger_positive / max(first_trigger_total, 1),
        "triggered_traces": float(first_trigger_total),
    }


def threshold_metrics(rows: list[dict[str, Any]], y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = scores >= threshold
    pos = y == 1
    neg = y == 0
    tp = int((pred & pos).sum())
    fp = int((pred & neg).sum())
    fn = int((~pred & pos).sum())
    tn = int((~pred & neg).sum())
    out = {
        "threshold": float(threshold),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }
    out.update(trace_stop_rates(rows, scores, threshold))
    return out


def choose_precision_thresholds(
    rows: list[dict[str, Any]],
    y: np.ndarray,
    scores: np.ndarray,
    precision_targets: list[float],
) -> list[dict[str, float]]:
    candidates = np.unique(scores)
    if len(candidates) > 3000:
        candidates = np.quantile(scores, np.linspace(0.0, 1.0, 3000))
    table: list[dict[str, float]] = []
    for target in precision_targets:
        best: dict[str, float] | None = None
        for tau in candidates:
            item = threshold_metrics(rows, y, scores, float(tau))
            if item["precision"] + 1e-12 < target:
                continue
            if best is None or (item["recall"], -item["fpr"], item["threshold"]) > (
                best["recall"],
                -best["fpr"],
                best["threshold"],
            ):
                best = item
        if best is None:
            best = threshold_metrics(rows, y, scores, float(scores.max()))
        best = dict(best)
        best["precision_target"] = float(target)
        table.append(best)
    return table


def write_score_details(path: Path, rows: list[dict[str, Any]], raw_scores: np.ndarray, calibrated_scores: np.ndarray) -> None:
    items = []
    for row, raw, cal in zip(rows, raw_scores, calibrated_scores, strict=True):
        items.append(
            {
                "trace_id": row.get("trace_id"),
                "problem_id": row.get("problem_id"),
                "step_index": int(row.get("step_index", 0)),
                "label": int(row.get("label", 0)),
                "trace_final_correct": bool(row.get("trace_final_correct", False)),
                "example_type": row.get("example_type"),
                "validity": row.get("validity"),
                "sample_weight": float(row.get("sample_weight", 1.0)),
                "raw_score": float(raw),
                "calibrated_score": float(cal),
                "step_text": row.get("step_text"),
            }
        )
    items.sort(key=lambda x: x["calibrated_score"], reverse=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def apply_weight_overrides(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    for row in rows:
        typ = str(row.get("example_type", ""))
        if typ == "fhis_positive" and args.positive_sample_weight is not None:
            row["sample_weight"] = float(args.positive_sample_weight)
        elif typ == "hard_fhis_false_negative" and args.hard_positive_weight is not None:
            row["sample_weight"] = float(args.hard_positive_weight)
        elif typ == "mined_hard_negative" and args.hard_negative_weight is not None:
            row["sample_weight"] = float(args.hard_negative_weight)
        elif typ == "strong_correct_negative" and args.correct_negative_weight is not None:
            row["sample_weight"] = float(args.correct_negative_weight)
        elif typ == "strong_prefhis_negative" and args.prefhis_negative_weight is not None:
            row["sample_weight"] = float(args.prefhis_negative_weight)


def score_report(name: str, rows: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    y = labels(rows)
    report: dict[str, Any] = {
        "name": name,
        "rows": len(rows),
        "positives": int(y.sum()),
        "negatives": int(len(y) - y.sum()),
        "ranking_metrics": evaluate_scores(rows, y, scores),
        "fixed_thresholds": [threshold_metrics(rows, y, scores, tau) for tau in [0.5, 0.7, 0.8, 0.9]],
        "precision_thresholds": choose_precision_thresholds(rows, y, scores, [0.70, 0.80, 0.85, 0.90, 0.95]),
        "by_example_type": dict(Counter(str(row.get("example_type", "unknown")) for row in rows)),
    }
    if len(np.unique(y)) == 2:
        report["auroc"] = float(roc_auc_score(y, scores))
        report["auprc"] = float(average_precision_score(y, scores))
        report["brier"] = float(brier_score_loss(y, np.clip(scores, 0.0, 1.0)))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train calibrated high-precision bad-step detector v2.1.")
    parser.add_argument("--config", default="data_generation/qwen25_fhis/configs/probe.yaml")
    parser.add_argument("--features", default="data_generation/qwen25_fhis/features/step_hidden_states_codex_clean.pt")
    parser.add_argument("--manifest", default="classifier/v2_runs/bad_step_v21/manifest.jsonl")
    parser.add_argument("--output-dir", default="classifier/v2_runs/bad_step_v21/detector_a")
    parser.add_argument("--positive-weight-multiplier", type=float, default=None)
    parser.add_argument("--ranking-weight", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--positive-sample-weight", type=float, default=None)
    parser.add_argument("--hard-positive-weight", type=float, default=None)
    parser.add_argument("--hard-negative-weight", type=float, default=None)
    parser.add_argument("--correct-negative-weight", type=float, default=None)
    parser.add_argument("--prefhis-negative-weight", type=float, default=None)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config.setdefault("probe_v2", {})
    cfg = config["probe_v2"]
    cfg.setdefault("use_scalars", True)
    cfg.setdefault("layer_embed_dim", 256)
    cfg.setdefault("mlp_dims", [256, 64])
    cfg.setdefault("dropout", 0.20)
    cfg.setdefault("lr", 2e-4)
    cfg.setdefault("weight_decay", 2e-4)
    cfg.setdefault("batch_size", 256)
    cfg.setdefault("max_epochs", 100)
    cfg.setdefault("patience", 14)
    cfg.setdefault("ranking_weight", 0.25)
    cfg.setdefault("ranking_margin", 1.0)
    cfg.setdefault("ranking_pairs_per_epoch", 4096)
    cfg.setdefault("monitor", "auprc")
    cfg.setdefault("positive_recall_target", 0.80)
    for key, value in {
        "positive_weight_multiplier": args.positive_weight_multiplier,
        "ranking_weight": args.ranking_weight,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }.items():
        if value is not None:
            cfg[key] = value

    payload = torch.load(args.features, map_location="cpu", weights_only=False)
    feature_rows = payload["rows"]
    manifest_rows = load_jsonl(Path(args.manifest))
    rows = rows_from_manifest(feature_rows, manifest_rows)
    apply_weight_overrides(rows, args)
    splits = split_rows(rows)
    train_rows, val_rows = inner_train_val(splits["train"])
    calibration_rows = splits["calibration"]
    hard_dev_rows = splits["hard_dev"]

    probe, train_info = train_layerwise_probe(train_rows, val_rows, config)
    cal_raw = probe.predict_scores(stack_features(calibration_rows), rows=calibration_rows)
    y_cal = labels(calibration_rows)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(cal_raw, y_cal)
    cal_iso = calibrator.predict(cal_raw)

    reports: dict[str, Any] = {
        "split_sizes": {
            "train_inner": len(train_rows),
            "val_inner": len(val_rows),
            "calibration": len(calibration_rows),
            "hard_dev": len(hard_dev_rows),
        },
        "train_info": train_info,
        "raw": {},
        "calibrated": {},
    }
    for name, split_rows_ in [("val_inner", val_rows), ("calibration", calibration_rows), ("hard_dev", hard_dev_rows)]:
        raw = probe.predict_scores(stack_features(split_rows_), rows=split_rows_)
        calibrated = calibrator.predict(raw)
        reports["raw"][name] = score_report(name, split_rows_, raw)
        reports["calibrated"][name] = score_report(name, split_rows_, calibrated)
        write_score_details(out / f"{name}_score_details.jsonl", split_rows_, raw, calibrated)

    reports["effective_probe_v2_config"] = cfg
    reports["weight_overrides"] = {
        "positive_sample_weight": args.positive_sample_weight,
        "hard_positive_weight": args.hard_positive_weight,
        "hard_negative_weight": args.hard_negative_weight,
        "correct_negative_weight": args.correct_negative_weight,
        "prefhis_negative_weight": args.prefhis_negative_weight,
    }

    frac_pos, mean_pred = calibration_curve(y_cal, cal_iso, n_bins=10, strategy="quantile")
    reports["calibration_curve"] = {
        "fraction_of_positives": [float(x) for x in frac_pos],
        "mean_predicted_value": [float(x) for x in mean_pred],
    }

    (out / "calibration_report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2))
    joblib.dump(
        {
            "model": probe,
            "calibrator": calibrator,
            "probe_kind": "bad_step_detector_v21_layerwise_mlp_isotonic",
            "config": config,
            "train_info": train_info,
            "calibration_report_path": str(out / "calibration_report.json"),
        },
        out / "probe_calibrated.joblib",
    )
    print(json.dumps(reports["split_sizes"], ensure_ascii=False, indent=2))
    print(json.dumps(reports["calibrated"]["hard_dev"]["precision_thresholds"], ensure_ascii=False, indent=2))
    print(f"Wrote {out / 'probe_calibrated.joblib'}")
    print(f"Wrote {out / 'calibration_report.json'}")


if __name__ == "__main__":
    main()
