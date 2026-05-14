from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fhis.io import append_jsonl, read_jsonl


FEATURE_NAMES = [
    "vote_count",
    "vote_frac",
    "num_answer_groups",
    "num_parseable_answers",
    "answer_len",
    "min_sample_index",
    "mean_sample_index",
    "min_max_score",
    "mean_max_score",
    "max_max_score",
    "min_mean_score",
    "mean_mean_score",
    "max_mean_score",
    "min_top2_mean_score",
    "mean_top2_mean_score",
    "max_top2_mean_score",
    "mean_num_steps",
    "min_num_steps",
    "max_num_steps",
    "parse_ok_frac",
    "risk_advantage_mean",
    "risk_advantage_max",
    "vote_margin",
    "vote_rank",
    "risk_rank_mean",
    "risk_rank_max",
]


@dataclass(frozen=True)
class AnswerGroup:
    problem_id: str
    canonical_answer: str
    answer: str | None
    label: int | None
    features: list[float]
    sample_indices: list[int]


def _finite(value: Any, default: float = 1.0) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _agg(values: list[float], fn: str, default: float = 1.0) -> float:
    if not values:
        return default
    if fn == "min":
        return min(values)
    if fn == "max":
        return max(values)
    if fn == "mean":
        return sum(values) / len(values)
    raise ValueError(fn)


def _rank(values: dict[str, float], key: str, reverse: bool = False) -> float:
    ordered = sorted(values.items(), key=lambda kv: (kv[1], kv[0]), reverse=reverse)
    for index, (candidate_key, _) in enumerate(ordered, start=1):
        if candidate_key == key:
            return float(index)
    return float(len(values) + 1)


def build_answer_groups(row: dict[str, Any]) -> list[AnswerGroup]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    first_answer: dict[str, str | None] = {}
    for sample in row.get("samples", []):
        canonical = sample.get("canonical_answer")
        if not canonical:
            continue
        canonical = str(canonical)
        grouped[canonical].append(sample)
        first_answer.setdefault(canonical, sample.get("answer"))

    if not grouped:
        return []

    num_parseable = sum(len(samples) for samples in grouped.values())
    vote_counts = {answer: len(samples) for answer, samples in grouped.items()}
    max_vote = max(vote_counts.values())
    vote_count_ranks = {
        answer: _rank(vote_counts, answer, reverse=True) for answer in grouped
    }

    group_min_mean: dict[str, float] = {}
    group_min_max: dict[str, float] = {}
    for answer, samples in grouped.items():
        mean_scores = [
            _finite(sample.get("probe_risk", {}).get("mean_score"))
            for sample in samples
        ]
        max_scores = [
            _finite(sample.get("probe_risk", {}).get("max_score"))
            for sample in samples
        ]
        group_min_mean[answer] = _agg(mean_scores, "min")
        group_min_max[answer] = _agg(max_scores, "min")

    best_min_mean = min(group_min_mean.values())
    best_min_max = min(group_min_max.values())

    groups: list[AnswerGroup] = []
    for answer, samples in grouped.items():
        sample_indices = [int(sample.get("sample_index", 999)) for sample in samples]
        max_scores = [_finite(sample.get("probe_risk", {}).get("max_score")) for sample in samples]
        mean_scores = [
            _finite(sample.get("probe_risk", {}).get("mean_score")) for sample in samples
        ]
        top2_scores = [
            _finite(sample.get("probe_risk", {}).get("top2_mean_score")) for sample in samples
        ]
        num_steps = [
            _finite(sample.get("probe_risk", {}).get("num_steps"), default=0.0)
            for sample in samples
        ]
        parse_ok = [
            1.0 if sample.get("probe_risk", {}).get("parse_ok") is True else 0.0
            for sample in samples
        ]
        label_values = [sample.get("rough_correct") for sample in samples]
        label = 1 if any(value is True for value in label_values) else 0
        if all(value is None for value in label_values):
            label = None

        vote_count = len(samples)
        features = [
            float(vote_count),
            vote_count / num_parseable if num_parseable else 0.0,
            float(len(grouped)),
            float(num_parseable),
            float(len(answer)),
            float(min(sample_indices)),
            sum(sample_indices) / len(sample_indices),
            _agg(max_scores, "min"),
            _agg(max_scores, "mean"),
            _agg(max_scores, "max"),
            _agg(mean_scores, "min"),
            _agg(mean_scores, "mean"),
            _agg(mean_scores, "max"),
            _agg(top2_scores, "min"),
            _agg(top2_scores, "mean"),
            _agg(top2_scores, "max"),
            _agg(num_steps, "mean", default=0.0),
            _agg(num_steps, "min", default=0.0),
            _agg(num_steps, "max", default=0.0),
            _agg(parse_ok, "mean", default=0.0),
            group_min_mean[answer] - best_min_mean,
            group_min_max[answer] - best_min_max,
            float(vote_count - max(v for key, v in vote_counts.items() if key != answer))
            if len(vote_counts) > 1
            else float(vote_count),
            vote_count_ranks[answer],
            _rank(group_min_mean, answer),
            _rank(group_min_max, answer),
        ]
        groups.append(
            AnswerGroup(
                problem_id=str(row.get("problem_id")),
                canonical_answer=answer,
                answer=first_answer.get(answer),
                label=label,
                features=features,
                sample_indices=sample_indices,
            )
        )
    return groups


def build_dataset(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[AnswerGroup]]:
    groups = [group for row in rows for group in build_answer_groups(row)]
    labeled = [group for group in groups if group.label is not None]
    x = np.asarray([group.features for group in labeled], dtype=np.float32)
    y = np.asarray([group.label for group in labeled], dtype=np.int64)
    return x, y, labeled


def choose_model(kind: str) -> Any:
    if kind == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
        )
    if kind == "forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=20260509,
            n_jobs=-1,
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=200,
            max_leaf_nodes=15,
            l2_regularization=0.1,
            random_state=20260509,
        )
    raise ValueError(f"unknown selector model: {kind}")


def predict_group_scores(model: Any, groups: list[AnswerGroup]) -> list[float]:
    if not groups:
        return []
    x = np.asarray([group.features for group in groups], dtype=np.float32)
    proba = model.predict_proba(x)
    return [float(score) for score in proba[:, 1]]


def select_for_row(model: Any, row: dict[str, Any]) -> dict[str, Any]:
    groups = build_answer_groups(row)
    if not groups:
        return {
            "problem_id": row.get("problem_id"),
            "status": "abstained",
            "selected_answer": None,
            "selected_canonical_answer": None,
            "selector_score": None,
            "rough_correct": None,
            "num_answer_groups": 0,
        }

    scores = predict_group_scores(model, groups)
    best_index = max(
        range(len(groups)),
        key=lambda index: (scores[index], -min(groups[index].sample_indices)),
    )
    best = groups[best_index]
    return {
        "problem_id": row.get("problem_id"),
        "status": "accepted",
        "selected_answer": best.answer,
        "selected_canonical_answer": best.canonical_answer,
        "selector_score": scores[best_index],
        "rough_correct": None if best.label is None else bool(best.label),
        "num_answer_groups": len(groups),
        "sample_indices": best.sample_indices,
        "candidates": [
            {
                "canonical_answer": group.canonical_answer,
                "answer": group.answer,
                "score": score,
                "label": group.label,
                "sample_indices": group.sample_indices,
            }
            for group, score in sorted(
                zip(groups, scores, strict=True), key=lambda item: item[1], reverse=True
            )
        ],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("status") == "accepted"]
    correct = [row for row in accepted if row.get("rough_correct") is True]
    known = [row for row in accepted if row.get("rough_correct") is not None]
    return {
        "num_problems": len(rows),
        "accepted": len(accepted),
        "abstained": len(rows) - len(accepted),
        "rough_correct": len(correct),
        "rough_solve_rate_all": len(correct) / len(rows) if rows else None,
        "rough_solve_rate_answered": len(correct) / len(known) if known else None,
    }


def train(args: argparse.Namespace) -> None:
    train_rows = list(read_jsonl(args.train))
    x_train, y_train, groups = build_dataset(train_rows)
    if len(set(y_train.tolist())) < 2:
        raise ValueError("training data must contain both correct and incorrect answer groups")

    model = choose_model(args.model)
    model.fit(x_train, y_train)

    payload = {
        "model": model,
        "feature_names": FEATURE_NAMES,
        "model_kind": args.model,
        "train_path": args.train,
    }
    Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.output_model)

    train_scores = model.predict_proba(x_train)[:, 1]
    report = {
        "model": args.model,
        "train_rows": len(train_rows),
        "train_answer_groups": len(groups),
        "positive_groups": int(y_train.sum()),
        "negative_groups": int((1 - y_train).sum()),
        "group_roc_auc": float(roc_auc_score(y_train, train_scores)),
        "group_average_precision": float(average_precision_score(y_train, train_scores)),
    }
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _fit_model(kind: str, rows: list[dict[str, Any]]) -> Any:
    x, y, _ = build_dataset(rows)
    if len(set(y.tolist())) < 2:
        raise ValueError(f"{kind}: data must contain both correct and incorrect answer groups")
    model = choose_model(kind)
    model.fit(x, y)
    return model


def _evaluate_model_on_rows(model: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [select_for_row(model, row) for row in rows]
    summary = summarize(selected)
    summary["oracle_any_sample_correct"] = sum(
        any(sample.get("rough_correct") is True for sample in row.get("samples", []))
        for row in rows
    )
    summary["majority_correct"] = sum(row.get("rough_correct") is True for row in rows)
    return summary


def train_validated(args: argparse.Namespace) -> None:
    rows = list(read_jsonl(args.train))
    train_rows, val_rows = train_test_split(
        rows,
        test_size=args.val_frac,
        random_state=args.seed,
        shuffle=True,
    )

    reports = []
    best_kind = None
    best_key = None
    for kind in args.models.split(","):
        kind = kind.strip()
        if not kind:
            continue
        model = _fit_model(kind, train_rows)
        train_summary = _evaluate_model_on_rows(model, train_rows)
        val_summary = _evaluate_model_on_rows(model, val_rows)
        report = {
            "model": kind,
            "train_summary": train_summary,
            "val_summary": val_summary,
        }
        reports.append(report)
        key = (
            val_summary["rough_correct"],
            val_summary["rough_solve_rate_all"] or 0.0,
            -val_summary["abstained"],
        )
        if best_key is None or key > best_key:
            best_key = key
            best_kind = kind

    if best_kind is None:
        raise ValueError("no selector model candidates were provided")

    final_model = _fit_model(best_kind, rows)
    final_summary = _evaluate_model_on_rows(final_model, rows)
    payload = {
        "model": final_model,
        "feature_names": FEATURE_NAMES,
        "model_kind": best_kind,
        "train_path": args.train,
        "selection": {
            "seed": args.seed,
            "val_frac": args.val_frac,
            "candidate_models": args.models,
            "reports": reports,
            "selected_model": best_kind,
            "full_dev_summary": final_summary,
        },
    }
    Path(args.output_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.output_model)
    report = payload["selection"]
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def apply(args: argparse.Namespace) -> None:
    payload = joblib.load(args.model_path)
    model = payload["model"]
    rows = list(read_jsonl(args.input))
    if Path(args.output).exists():
        Path(args.output).unlink()
    out_rows = []
    for row in rows:
        selected = select_for_row(model, row)
        append_jsonl(args.output, [selected])
        out_rows.append(selected)
    summary = summarize(out_rows)
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/apply an answer-level selector on probe-scored candidate pools.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument("--train", required=True)
    train_parser.add_argument("--output-model", required=True)
    train_parser.add_argument("--model", choices=["logistic", "forest", "hgb"], default="logistic")
    train_parser.add_argument("--report", default=None)
    train_parser.set_defaults(func=train)

    validated_parser = sub.add_parser("train-validated")
    validated_parser.add_argument("--train", required=True)
    validated_parser.add_argument("--output-model", required=True)
    validated_parser.add_argument("--models", default="logistic,forest,hgb")
    validated_parser.add_argument("--val-frac", type=float, default=0.35)
    validated_parser.add_argument("--seed", type=int, default=20260509)
    validated_parser.add_argument("--report", default=None)
    validated_parser.set_defaults(func=train_validated)

    apply_parser = sub.add_parser("apply")
    apply_parser.add_argument("--model-path", required=True)
    apply_parser.add_argument("--input", required=True)
    apply_parser.add_argument("--output", required=True)
    apply_parser.add_argument("--summary", default=None)
    apply_parser.set_defaults(func=apply)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
