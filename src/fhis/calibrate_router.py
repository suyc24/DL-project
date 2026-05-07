from __future__ import annotations

import argparse
import json
from pathlib import Path

from fhis.config import load_config


def threshold_for_rate(scores: np.ndarray, target_rate: float) -> float:
    import numpy as np

    if len(scores) == 0:
        return 0.5
    target_rate = min(max(float(target_rate), 0.0), 1.0)
    if target_rate <= 0:
        return float(np.nextafter(scores.max(), np.inf))
    if target_rate >= 1:
        return float(np.nextafter(scores.min(), -np.inf))
    return float(np.quantile(scores, 1.0 - target_rate, method="higher"))


def threshold_for_positive_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    positive_recall: float,
) -> float:
    import numpy as np

    positive_scores = scores[y_true == 1]
    if len(positive_scores) == 0:
        return 0.5
    positive_recall = min(max(float(positive_recall), 0.0), 1.0)
    return float(np.quantile(positive_scores, 1.0 - positive_recall, method="lower"))


def rate_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    import numpy as np

    routed = scores >= threshold
    positives = y_true == 1
    negatives = y_true == 0
    return {
        "threshold": float(threshold),
        "verification_rate": float(routed.mean()) if len(routed) else float("nan"),
        "positive_recall": float((routed & positives).sum() / max(positives.sum(), 1)),
        "negative_skip_rate": float(((~routed) & negatives).sum() / max(negatives.sum(), 1)),
        "num_steps": float(len(scores)),
        "num_positive_steps": float(positives.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate online router thresholds on validation rows.")
    parser.add_argument("--config", default="data_generation/qwen25_fhis/configs/probe.yaml")
    parser.add_argument("--features", default=None)
    parser.add_argument("--probe-model", default=None)
    parser.add_argument("--target-rate", type=float, default=0.30)
    parser.add_argument("--positive-recall", type=float, default=None)
    parser.add_argument(
        "--output",
        default="data_generation/qwen25_fhis/results/router_threshold.json",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    import joblib
    import torch

    from fhis.train_probe import labels, rows_for_split, score_probe, split_problem_ids

    features_path = args.features or config["paths"]["hidden_states"]
    probe_path = args.probe_model or config["paths"]["probe_model"]
    payload = torch.load(features_path, map_location="cpu", weights_only=False)
    rows = payload["rows"]
    splits = split_problem_ids(
        rows,
        train_frac=float(config["split"]["train"]),
        val_frac=float(config["split"]["val"]),
        seed=int(config.get("seed", 0)),
    )
    val_rows = rows_for_split(rows, splits["val"])
    probe_payload = joblib.load(probe_path)
    probe = probe_payload["model"] if isinstance(probe_payload, dict) else probe_payload
    scores = score_probe(probe, val_rows)
    y_val = labels(val_rows)

    by_rate = threshold_for_rate(scores, args.target_rate)
    payload_out = {
        "target_rate": args.target_rate,
        "by_target_rate": rate_metrics(y_val, scores, by_rate),
    }
    if args.positive_recall is not None:
        by_recall = threshold_for_positive_recall(y_val, scores, args.positive_recall)
        payload_out["positive_recall_target"] = args.positive_recall
        payload_out["by_positive_recall"] = rate_metrics(y_val, scores, by_recall)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload_out, indent=2), encoding="utf-8")
    print(json.dumps(payload_out, indent=2))


if __name__ == "__main__":
    main()
