from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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


def default_device(config: dict[str, Any]) -> str:
    requested = str(config["probe"].get("device", "auto"))
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


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


class HiddenStateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, int(hidden_dim)),
                    nn.LayerNorm(int(hidden_dim)),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            prev = int(hidden_dim)
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RecallBiasedLoss(nn.Module):
    """BCE plus soft Tversky/margin terms that make false negatives expensive."""

    def __init__(
        self,
        pos_weight: torch.Tensor,
        bce_weight: float,
        tversky_weight: float,
        tversky_alpha: float,
        tversky_beta: float,
        positive_margin_weight: float,
        positive_logit_margin: float,
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.bce_weight = float(bce_weight)
        self.tversky_weight = float(tversky_weight)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.positive_margin_weight = float(positive_margin_weight)
        self.positive_logit_margin = float(positive_logit_margin)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        targets = targets.float()
        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets)).sum()
        fn = ((1.0 - probs) * targets).sum()
        tversky = 1.0 - (tp + 1e-6) / (
            tp + self.tversky_alpha * fp + self.tversky_beta * fn + 1e-6
        )
        if targets.sum() > 0:
            positive_logits = logits[targets > 0.5]
            margin = torch.relu(self.positive_logit_margin - positive_logits).pow(2).mean()
        else:
            margin = logits.new_tensor(0.0)
        return (
            self.bce_weight * bce
            + self.tversky_weight * tversky
            + self.positive_margin_weight * margin
        )


class TorchMLPProbe:
    """Serializable predictor wrapper with sklearn-like predict_proba."""

    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        state_dict: dict[str, torch.Tensor],
        hidden_dims: list[int],
        dropout: float,
        decision_threshold: float = 0.5,
    ) -> None:
        self.mean = mean.astype(np.float32)
        self.scale = scale.astype(np.float32)
        self.hidden_dims = [int(dim) for dim in hidden_dims]
        self.dropout = float(dropout)
        self.decision_threshold = float(decision_threshold)
        self.model = HiddenStateMLP(self.mean.shape[0], self.hidden_dims, self.dropout)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return ((x.astype(np.float32) - self.mean) / self.scale).astype(np.float32)

    def predict_scores(self, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        x_std = self._standardize(x)
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x_std), batch_size):
                xb = torch.from_numpy(x_std[start : start + batch_size])
                logits = self.model(xb)
                scores.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(scores) if scores else np.asarray([], dtype=np.float32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = self.predict_scores(x)
        return np.column_stack([1.0 - scores, scores])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_scores(x) >= self.decision_threshold).astype(np.int64)


def choose_recall_biased_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    positive_recall_target: float,
    negative_accuracy_floor: float,
) -> float:
    """Choose the highest-specificity threshold under a positive-recall constraint."""
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0 or len(np.unique(y_true)) < 2:
        return 0.5

    candidates = np.unique(scores)
    candidates = np.concatenate(
        [
            [float(np.nextafter(candidates.min(), -np.inf))],
            candidates,
            [float(np.nextafter(candidates.max(), np.inf))],
        ]
    )

    best: tuple[float, float, float] | None = None
    fallback: tuple[float, float, float] | None = None
    for threshold in candidates:
        pred = scores >= threshold
        positives = y_true == 1
        negatives = y_true == 0
        pos_recall = float((pred & positives).sum() / max(positives.sum(), 1))
        neg_acc = float(((~pred) & negatives).sum() / max(negatives.sum(), 1))
        item = (neg_acc, pos_recall, float(threshold))
        if pos_recall >= positive_recall_target:
            if fallback is None or item > fallback:
                fallback = item
            if neg_acc >= negative_accuracy_floor and (best is None or item > best):
                best = item
    chosen = best or fallback
    return chosen[2] if chosen is not None else 0.5


def fit_mlp_probe(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    config: dict[str, Any],
    feature_slice: slice | None = None,
) -> TorchMLPProbe:
    probe_cfg = config["probe"]
    mlp_cfg = probe_cfg.get("mlp", {})
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train = stack_features(train_rows)
    x_val = stack_features(val_rows if val_rows else train_rows)
    if feature_slice is not None:
        x_train = x_train[:, feature_slice]
        x_val = x_val[:, feature_slice]
    y_train = labels(train_rows).astype(np.float32)
    y_val = labels(val_rows if val_rows else train_rows)

    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    x_train = ((x_train.astype(np.float32) - mean) / scale).astype(np.float32)
    x_val = ((x_val.astype(np.float32) - mean) / scale).astype(np.float32)

    device = torch.device(default_device(config))
    hidden_dims = [int(dim) for dim in mlp_cfg.get("hidden_dims", [512, 128])]
    dropout = float(mlp_cfg.get("dropout", 0.2))
    model = HiddenStateMLP(x_train.shape[1], hidden_dims, dropout).to(device)

    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight_value = (negatives / max(positives, 1.0)) * float(
        mlp_cfg.get("positive_weight_multiplier", 1.0)
    )
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    loss_kind = str(mlp_cfg.get("loss", "weighted_bce")).lower()
    if loss_kind == "recall_tversky":
        loss_fn = RecallBiasedLoss(
            pos_weight=pos_weight,
            bce_weight=float(mlp_cfg.get("bce_weight", 1.0)),
            tversky_weight=float(mlp_cfg.get("tversky_weight", 0.5)),
            tversky_alpha=float(mlp_cfg.get("tversky_alpha", 0.2)),
            tversky_beta=float(mlp_cfg.get("tversky_beta", 0.8)),
            positive_margin_weight=float(mlp_cfg.get("positive_margin_weight", 0.0)),
            positive_logit_margin=float(mlp_cfg.get("positive_logit_margin", 0.0)),
        )
    else:
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight if bool(mlp_cfg.get("use_pos_weight", True)) else None
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(mlp_cfg.get("lr", 3e-4)),
        weight_decay=float(mlp_cfg.get("weight_decay", 1e-4)),
    )

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        train_ds,
        batch_size=int(mlp_cfg.get("batch_size", 256)),
        shuffle=True,
        generator=generator,
    )

    max_epochs = int(mlp_cfg.get("max_epochs", 120))
    patience = int(mlp_cfg.get("patience", 15))
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    val_x_tensor = torch.from_numpy(x_val).to(device)

    for _epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(mlp_cfg.get("grad_clip", 1.0)))
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(val_x_tensor)).cpu().numpy()
        val_metrics = evaluate_scores(val_rows if val_rows else train_rows, y_val, val_scores)
        monitor = str(mlp_cfg.get("monitor", "auprc"))
        score = float(val_metrics.get(monitor, float("nan")))
        if not np.isfinite(score):
            score = -float(loss.detach().cpu())
        if score > best_score + float(mlp_cfg.get("min_delta", 1e-4)):
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    probe = TorchMLPProbe(mean, scale, best_state, hidden_dims, dropout)
    val_scores = probe.predict_scores(x_val * scale + mean)
    probe.decision_threshold = choose_recall_biased_threshold(
        y_val,
        val_scores,
        positive_recall_target=float(mlp_cfg.get("positive_recall_target", 0.99)),
        negative_accuracy_floor=float(mlp_cfg.get("negative_accuracy_floor", 0.80)),
    )
    return probe


def fit_hidden_probe(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    config: dict[str, Any],
    feature_slice: slice | None = None,
) -> Any:
    kind = str(config["probe"].get("kind", "logistic")).lower()
    if kind == "logistic":
        return fit_logistic_probe(train_rows, config, feature_slice=feature_slice)
    if kind == "mlp":
        return fit_mlp_probe(train_rows, val_rows, config, feature_slice=feature_slice)
    raise ValueError(f"Unsupported probe.kind={kind!r}")


def score_probe(
    probe: Any,
    rows: list[dict[str, Any]],
    feature_slice: slice | None = None,
) -> np.ndarray:
    x = stack_features(rows)
    if feature_slice is not None:
        x = x[:, feature_slice]
    return probe.predict_proba(x)[:, 1]


def train_hidden_probe(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    config: dict[str, Any],
    feature_slice: slice | None = None,
) -> np.ndarray:
    probe = fit_hidden_probe(train_rows, val_rows, config, feature_slice=feature_slice)
    return score_probe(probe, test_rows, feature_slice=feature_slice)


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


def threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_pred_05 = (scores >= 0.5).astype(np.int64)
    y_pred = (scores >= threshold).astype(np.int64)
    positives = y_true == 1
    negatives = y_true == 0
    positive_accuracy = float((y_pred[positives] == 1).mean()) if positives.any() else float("nan")
    negative_accuracy = float((y_pred[negatives] == 0).mean()) if negatives.any() else float("nan")
    return {
        "decision_threshold": float(threshold),
        "accuracy_at_0_5": float(accuracy_score(y_true, y_pred_05)),
        "accuracy_at_threshold": float(accuracy_score(y_true, y_pred)),
        "positive_accuracy_at_threshold": positive_accuracy,
        "negative_accuracy_at_threshold": negative_accuracy,
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y_true, y_pred_05)),
        "balanced_accuracy_at_threshold": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_at_0_5": float(f1_score(y_true, y_pred_05, zero_division=0)),
        "f1_at_threshold": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_probe_scores(
    rows: list[dict[str, Any]],
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    metrics = evaluate_scores(rows, y_true, scores)
    metrics.update(threshold_metrics(y_true, scores, threshold=threshold))
    return metrics


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
        metrics = evaluate_probe_scores(test_rows, y_test, scores)
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

    payload = torch.load(features_path, map_location="cpu", weights_only=False)
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

    probe_kind = str(config["probe"].get("kind", "logistic")).lower()
    hidden_name = f"hidden_{probe_kind}"
    hidden_probe = fit_hidden_probe(train_rows, val_rows, config)
    all_scores = {
        hidden_name: score_probe(hidden_probe, eval_rows),
        "hidden_logistic": train_logistic_probe(train_rows, eval_rows, config),
        **score_baselines(train_rows, eval_rows, int(config.get("seed", 0)), config),
    }
    if hidden_name == "hidden_logistic":
        all_scores.pop("hidden_logistic")
    thresholds = {hidden_name: float(getattr(hidden_probe, "decision_threshold", 0.5))}
    metrics = {
        name: evaluate_probe_scores(
            eval_rows,
            y_eval,
            scores,
            threshold=thresholds.get(name, 0.5),
        )
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
        if probe_kind == "logistic":
            probe = fit_logistic_probe(train_rows + val_rows, config)
        else:
            probe = hidden_probe
        joblib.dump(
            {
                "model": probe,
                "probe_kind": probe_kind,
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
