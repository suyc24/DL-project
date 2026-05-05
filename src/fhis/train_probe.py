from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fhis.config import load_config
from fhis.metrics import evaluate_scores


def split_problem_ids(
    rows: list[dict[str, Any]],
    train_frac: float,
    val_frac: float,
    seed: int,
) -> dict[str, set[str]]:
    problem_ids = sorted({str(row["problem_id"]) for row in rows})
    rng = random.Random(seed)
    rng.shuffle(problem_ids)
    n = len(problem_ids)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    train = set(problem_ids[:n_train])
    val = set(problem_ids[n_train : n_train + n_val])
    test = set(problem_ids[n_train + n_val :])
    return {"train": train, "val": val, "test": test}


def rows_for_split(rows: list[dict[str, Any]], ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row["problem_id"]) in ids]


def stack_features(rows: list[dict[str, Any]]) -> np.ndarray:
    return torch.stack([row["feature"].float().cpu() for row in rows]).numpy()


def labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def finite_baseline(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if np.isfinite(arr).any():
        fill = float(np.nanmean(arr[np.isfinite(arr)]))
    else:
        fill = 0.0
    arr[~np.isfinite(arr)] = fill
    return arr


def fit_logistic_probe(
    train_rows: list[dict[str, Any]],
    config: dict[str, Any],
    feature_slice: slice | None = None,
) -> Any:
    x_train = stack_features(train_rows)
    if feature_slice is not None:
        x_train = x_train[:, feature_slice]
    y_train = labels(train_rows)
    probe_cfg = config["probe"]
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=int(probe_cfg.get("max_iter", 2000)),
            class_weight=probe_cfg.get("class_weight", "balanced"),
            solver="lbfgs",
        ),
    )
    clf.fit(x_train, y_train)
    return clf


def train_logistic_probe(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    config: dict[str, Any],
    feature_slice: slice | None = None,
) -> np.ndarray:
    clf = fit_logistic_probe(train_rows, config, feature_slice=feature_slice)
    x_test = stack_features(test_rows)
    if feature_slice is not None:
        x_test = x_test[:, feature_slice]
    return clf.predict_proba(x_test)[:, 1]


def score_baselines(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores: dict[str, np.ndarray] = {
        "random": rng.random(len(test_rows)),
        "step_index": finite_baseline([row["baselines"]["step_index"] for row in test_rows]),
        "step_length": finite_baseline(
            [row["baselines"]["step_length_chars"] for row in test_rows]
        ),
        "low_mean_token_logprob": -finite_baseline(
            [row["baselines"]["mean_token_logprob"] for row in test_rows]
        ),
    }

    text_model = make_pipeline(
        TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=50000),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    text_model.fit([row.get("step_text", "") for row in train_rows], labels(train_rows))
    scores["text_tfidf_logistic"] = text_model.predict_proba(
        [row.get("step_text", "") for row in test_rows]
    )[:, 1]

    wrong_y = np.asarray([0 if row.get("trace_final_correct") else 1 for row in train_rows])
    if len(np.unique(wrong_y)) == 2:
        final_probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=int(config["probe"].get("max_iter", 2000)),
                class_weight="balanced",
            ),
        )
        final_probe.fit(stack_features(train_rows), wrong_y)
        scores["final_wrongness_probe"] = final_probe.predict_proba(stack_features(test_rows))[:, 1]

    return scores


def layer_sweep(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if not train_rows:
        return []
    layer_ids = train_rows[0]["layer_ids"]
    feature_dim = int(train_rows[0]["feature"].numel())
    hidden_size = feature_dim // len(layer_ids)
    rows: list[dict[str, Any]] = []
    y_test = labels(test_rows)
    for pos, layer_id in enumerate(layer_ids):
        feature_slice = slice(pos * hidden_size, (pos + 1) * hidden_size)
        scores = train_logistic_probe(train_rows, test_rows, config, feature_slice=feature_slice)
        metrics = evaluate_scores(test_rows, y_test, scores)
        rows.append({"layer": layer_id, **metrics})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FHIS probes and baselines.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--features", default=None)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--layer-output", default=None)
    parser.add_argument("--probe-output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    features_path = args.features or config["paths"]["hidden_states"]
    metrics_path = Path(args.metrics_output or config["paths"]["metrics"])
    layer_path = Path(args.layer_output or config["paths"]["layer_sweep"])
    probe_output = args.probe_output or config["paths"].get("probe_model")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    layer_path.parent.mkdir(parents=True, exist_ok=True)

    payload = torch.load(features_path, map_location="cpu")
    rows = payload["rows"]
    split_cfg = config["split"]
    splits = split_problem_ids(
        rows,
        train_frac=float(split_cfg["train"]),
        val_frac=float(split_cfg["val"]),
        seed=int(config.get("seed", 0)),
    )
    train_rows = rows_for_split(rows, splits["train"])
    val_rows = rows_for_split(rows, splits["val"])
    test_rows = rows_for_split(rows, splits["test"])
    eval_rows = test_rows if test_rows else val_rows
    y_eval = labels(eval_rows)

    all_scores = {
        "hidden_logistic": train_logistic_probe(train_rows, eval_rows, config),
        **score_baselines(train_rows, eval_rows, int(config.get("seed", 0)), config),
    }
    metrics = {
        name: evaluate_scores(eval_rows, y_eval, scores)
        for name, scores in all_scores.items()
    }
    metrics["_split"] = {
        "train_steps": len(train_rows),
        "val_steps": len(val_rows),
        "test_steps": len(test_rows),
        "eval_split": "test" if test_rows else "val",
        "train_problems": len(splits["train"]),
        "val_problems": len(splits["val"]),
        "test_problems": len(splits["test"]),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    sweep_rows = layer_sweep(train_rows, eval_rows, config)
    pd.DataFrame(sweep_rows).to_csv(layer_path, index=False)

    if probe_output:
        probe_path = Path(probe_output)
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        probe = fit_logistic_probe(train_rows + val_rows, config)
        joblib.dump(
            {
                "model": probe,
                "feature_layers": rows[0]["layer_ids"] if rows else [],
                "feature_dim": int(rows[0]["feature"].numel()) if rows else 0,
                "trained_on": {
                    "train_steps": len(train_rows),
                    "val_steps": len(val_rows),
                    "train_problems": len(splits["train"]),
                    "val_problems": len(splits["val"]),
                },
                "config": config,
            },
            probe_path,
        )

    print(json.dumps(metrics, indent=2))
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote layer sweep to {layer_path}")
    if probe_output:
        print(f"Wrote probe model to {probe_output}")


if __name__ == "__main__":
    main()
