from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fhis.config import load_config
from fhis.metrics import evaluate_scores
from fhis.train_probe import labels, rows_for_split, split_problem_ids, stack_features


if __name__ == "__main__":
    sys.modules.setdefault("fhis.train_probe_v2", sys.modules[__name__])


def scalar_features(rows: list[dict[str, Any]], use_scalars: bool) -> np.ndarray:
    if not use_scalars:
        return np.zeros((len(rows), 0), dtype=np.float32)
    values: list[list[float]] = []
    for row in rows:
        baselines = row.get("baselines") or {}
        mean_logprob = baselines.get("mean_token_logprob")
        if mean_logprob is None or not np.isfinite(float(mean_logprob)):
            mean_logprob = 0.0
        values.append(
            [
                float(row.get("step_index", baselines.get("step_index", 0.0))),
                float(baselines.get("step_length_chars", len(str(row.get("step_text", ""))))),
                float(mean_logprob),
            ]
        )
    return np.asarray(values, dtype=np.float32)


def split_feature_tensor(x: np.ndarray, n_layers: int) -> tuple[np.ndarray, int]:
    if x.shape[1] % n_layers != 0:
        raise ValueError(f"feature_dim={x.shape[1]} is not divisible by n_layers={n_layers}")
    hidden_size = x.shape[1] // n_layers
    return x.reshape(x.shape[0], n_layers, hidden_size), hidden_size


def standardize(x: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - mean) / scale).astype(np.float32)


def mean_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[1] == 0:
        return np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


class LayerwiseProbeNet(nn.Module):
    def __init__(
        self,
        n_layers: int,
        hidden_size: int,
        scalar_dim: int,
        layer_embed_dim: int,
        mlp_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        self.hidden_size = int(hidden_size)
        self.scalar_dim = int(scalar_dim)
        self.layer_proj = nn.Sequential(
            nn.Linear(hidden_size, layer_embed_dim),
            nn.LayerNorm(layer_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layer_score = nn.Linear(layer_embed_dim, 1)

        dims = [layer_embed_dim + scalar_dim, *mlp_dims, 1]
        layers: list[nn.Module] = []
        for src, dst in zip(dims, dims[1:]):
            layers.append(nn.Linear(src, dst))
            if dst != 1:
                layers.extend([nn.LayerNorm(dst), nn.GELU(), nn.Dropout(dropout)])
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_layers: torch.Tensor, x_scalars: torch.Tensor) -> torch.Tensor:
        layer_emb = self.layer_proj(x_layers)
        weights = torch.softmax(self.layer_score(layer_emb).squeeze(-1), dim=-1)
        pooled = (layer_emb * weights.unsqueeze(-1)).sum(dim=1)
        features = torch.cat([pooled, x_scalars], dim=-1)
        return self.mlp(features).squeeze(-1)


@dataclass
class LayerwiseProbeArtifact:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    scalar_mean: np.ndarray
    scalar_scale: np.ndarray
    state_dict: dict[str, torch.Tensor]
    layer_ids: list[int]
    hidden_size: int
    layer_embed_dim: int
    mlp_dims: list[int]
    dropout: float
    decision_threshold: float

    def build_model(self) -> LayerwiseProbeNet:
        model = LayerwiseProbeNet(
            n_layers=len(self.layer_ids),
            hidden_size=self.hidden_size,
            scalar_dim=int(self.scalar_mean.shape[0]),
            layer_embed_dim=self.layer_embed_dim,
            mlp_dims=self.mlp_dims,
            dropout=self.dropout,
        )
        model.load_state_dict(self.state_dict)
        model.eval()
        return model

    def _prepare(self, x: np.ndarray, rows: list[dict[str, Any]] | None = None) -> tuple[np.ndarray, np.ndarray]:
        x_std = standardize(x, self.feature_mean, self.feature_scale)
        x_layers, _ = split_feature_tensor(x_std, len(self.layer_ids))
        if rows is None:
            scalars = np.zeros((x.shape[0], self.scalar_mean.shape[0]), dtype=np.float32)
        else:
            scalars = scalar_features(rows, use_scalars=self.scalar_mean.shape[0] > 0)
        scalars = standardize(scalars, self.scalar_mean, self.scalar_scale)
        return x_layers, scalars

    def predict_scores(
        self,
        x: np.ndarray,
        rows: list[dict[str, Any]] | None = None,
        batch_size: int = 1024,
    ) -> np.ndarray:
        x_layers, scalars = self._prepare(x, rows=rows)
        model = self.build_model()
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x_layers), batch_size):
                xb = torch.from_numpy(x_layers[start : start + batch_size])
                sb = torch.from_numpy(scalars[start : start + batch_size])
                logits = model(xb, sb)
                scores.append(torch.sigmoid(logits).numpy())
        return np.concatenate(scores) if scores else np.asarray([], dtype=np.float32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = self.predict_scores(x)
        return np.column_stack([1.0 - scores, scores])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_scores(x) >= self.decision_threshold).astype(np.int64)


LayerwiseProbeNet.__module__ = "fhis.train_probe_v2"
LayerwiseProbeArtifact.__module__ = "fhis.train_probe_v2"


def build_pair_indices(rows: list[dict[str, Any]]) -> np.ndarray:
    by_trace: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for idx, row in enumerate(rows):
        by_trace.setdefault(str(row["trace_id"]), []).append((idx, row))
    pairs: list[tuple[int, int]] = []
    for items in by_trace.values():
        pos = [idx for idx, row in items if int(row["label"]) == 1]
        neg = [idx for idx, row in items if int(row["label"]) == 0]
        if len(pos) != 1 or not neg:
            continue
        pairs.extend((pos[0], neg_idx) for neg_idx in neg)
    return np.asarray(pairs, dtype=np.int64)


def choose_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_recall: float,
    max_fpr: float | None,
) -> float:
    candidates = np.unique(scores)
    if len(candidates) == 0 or len(np.unique(y_true)) < 2:
        return 0.5
    positives = y_true == 1
    negatives = y_true == 0
    best: tuple[float, float, float] | None = None
    fallback: tuple[float, float, float] | None = None
    for tau in candidates:
        pred = scores >= tau
        recall = float((pred & positives).sum() / max(positives.sum(), 1))
        fpr = float((pred & negatives).sum() / max(negatives.sum(), 1))
        item = (-fpr, recall, float(tau))
        if recall >= target_recall:
            if fallback is None or item > fallback:
                fallback = item
            if max_fpr is None or fpr <= max_fpr:
                if best is None or item > best:
                    best = item
    chosen = best or fallback
    return chosen[2] if chosen is not None else 0.5


def training_sample_weights(
    rows: list[dict[str, Any]],
    y: np.ndarray,
    *,
    positive_weight: float,
    prefhis_negative_weight: float,
    correct_negative_weight: float,
) -> np.ndarray:
    weights = np.ones(len(rows), dtype=np.float32)
    for idx, row in enumerate(rows):
        if int(y[idx]) == 1:
            weights[idx] = float(positive_weight)
        elif bool(row.get("trace_final_correct", False)):
            weights[idx] = float(correct_negative_weight)
        else:
            weights[idx] = float(prefhis_negative_weight)
    return weights


def run_epoch(
    model: LayerwiseProbeNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    seen = 0
    for xb, sb, yb, wb in loader:
        xb = xb.to(device)
        sb = sb.to(device)
        yb = yb.to(device)
        wb = wb.to(device)
        optimizer.zero_grad(set_to_none=True)
        per_sample_loss = loss_fn(model(xb, sb), yb)
        loss = (per_sample_loss * wb).sum() / wb.sum().clamp_min(1.0)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach().cpu()) * len(yb)
        seen += len(yb)
    return total / max(seen, 1)


def ranking_step(
    model: LayerwiseProbeNet,
    x_layers: np.ndarray,
    scalars: np.ndarray,
    pairs: np.ndarray,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    margin: float,
    sample_size: int,
    rng: random.Random,
) -> float:
    if len(pairs) == 0 or sample_size <= 0:
        return 0.0
    if len(pairs) > sample_size:
        chosen = rng.sample(range(len(pairs)), sample_size)
        pairs = pairs[chosen]
    pos_idx = torch.from_numpy(pairs[:, 0]).long().to(device)
    neg_idx = torch.from_numpy(pairs[:, 1]).long().to(device)
    xb = torch.from_numpy(x_layers).to(device)
    sb = torch.from_numpy(scalars).to(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(xb, sb)
    loss = torch.relu(float(margin) - logits[pos_idx] + logits[neg_idx]).mean()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().cpu())


def score_model(
    model: LayerwiseProbeNet,
    x_layers: np.ndarray,
    scalars: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_layers), batch_size):
            xb = torch.from_numpy(x_layers[start : start + batch_size]).to(device)
            sb = torch.from_numpy(scalars[start : start + batch_size]).to(device)
            scores.append(torch.sigmoid(model(xb, sb)).cpu().numpy())
    return np.concatenate(scores) if scores else np.asarray([], dtype=np.float32)


def train_layerwise_probe(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[LayerwiseProbeArtifact, dict[str, Any]]:
    cfg = config.get("probe_v2", {})
    use_scalars = bool(cfg.get("use_scalars", False))
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    layer_ids = train_rows[0]["layer_ids"]
    x_train_flat = stack_features(train_rows).astype(np.float32)
    x_val_flat = stack_features(val_rows).astype(np.float32)
    feature_mean, feature_scale = mean_scale(x_train_flat)
    x_train_flat = standardize(x_train_flat, feature_mean, feature_scale)
    x_val_flat = standardize(x_val_flat, feature_mean, feature_scale)
    x_train, hidden_size = split_feature_tensor(x_train_flat, len(layer_ids))
    x_val, _ = split_feature_tensor(x_val_flat, len(layer_ids))

    s_train_raw = scalar_features(train_rows, use_scalars=use_scalars)
    s_val_raw = scalar_features(val_rows, use_scalars=use_scalars)
    scalar_mean, scalar_scale = mean_scale(s_train_raw)
    s_train = standardize(s_train_raw, scalar_mean, scalar_scale)
    s_val = standardize(s_val_raw, scalar_mean, scalar_scale)
    y_train = labels(train_rows).astype(np.float32)
    y_val = labels(val_rows)

    device = torch.device(str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    model = LayerwiseProbeNet(
        n_layers=len(layer_ids),
        hidden_size=hidden_size,
        scalar_dim=s_train.shape[1],
        layer_embed_dim=int(cfg.get("layer_embed_dim", 256)),
        mlp_dims=[int(x) for x in cfg.get("mlp_dims", [256, 64])],
        dropout=float(cfg.get("dropout", 0.15)),
    ).to(device)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor(
        [(neg / max(pos, 1.0)) * float(cfg.get("positive_weight_multiplier", 1.0))],
        dtype=torch.float32,
        device=device,
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(s_train),
            torch.from_numpy(y_train),
            torch.from_numpy(
                training_sample_weights(
                    train_rows,
                    y_train,
                    positive_weight=float(cfg.get("positive_sample_weight", 1.0)),
                    prefhis_negative_weight=float(cfg.get("prefhis_negative_weight", 1.0)),
                    correct_negative_weight=float(cfg.get("correct_negative_weight", 1.0)),
                )
            ),
        ),
        batch_size=int(cfg.get("batch_size", 256)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    pairs = build_pair_indices(train_rows)
    max_epochs = int(cfg.get("max_epochs", 80))
    patience = int(cfg.get("patience", 12))
    ranking_weight = float(cfg.get("ranking_weight", 0.5))
    ranking_margin = float(cfg.get("ranking_margin", 1.0))
    ranking_pairs_per_epoch = int(cfg.get("ranking_pairs_per_epoch", 4096))
    monitor = str(cfg.get("monitor", "auprc"))

    best_score = -float("inf")
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, loader, optimizer, loss_fn, device)
        rank_loss = 0.0
        if ranking_weight > 0.0:
            rank_loss = ranking_step(
                model,
                x_train,
                s_train,
                pairs,
                optimizer,
                device,
                ranking_margin,
                ranking_pairs_per_epoch,
                rng,
            )
        val_scores = score_model(model, x_val, s_val, device, int(cfg.get("eval_batch_size", 2048)))
        if len(np.unique(y_val)) == 2:
            val_auroc = float(roc_auc_score(y_val, val_scores))
            val_auprc = float(average_precision_score(y_val, val_scores))
        else:
            val_auroc = float("nan")
            val_auprc = float("nan")
        score = val_auprc if monitor == "auprc" else val_auroc
        history.append(
            {
                "epoch": epoch,
                "train_bce": train_loss,
                "ranking_loss": rank_loss,
                "val_auroc": val_auroc,
                "val_auprc": val_auprc,
            }
        )
        if np.isfinite(score) and score > best_score + float(cfg.get("min_delta", 1e-4)):
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    artifact = LayerwiseProbeArtifact(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        scalar_mean=scalar_mean,
        scalar_scale=scalar_scale,
        state_dict=best_state,
        layer_ids=[int(x) for x in layer_ids],
        hidden_size=int(hidden_size),
        layer_embed_dim=int(cfg.get("layer_embed_dim", 256)),
        mlp_dims=[int(x) for x in cfg.get("mlp_dims", [256, 64])],
        dropout=float(cfg.get("dropout", 0.15)),
        decision_threshold=0.5,
    )
    val_raw = stack_features(val_rows).astype(np.float32)
    val_scores = artifact.predict_scores(val_raw, rows=val_rows)
    artifact.decision_threshold = choose_threshold(
        y_val,
        val_scores,
        target_recall=float(cfg.get("positive_recall_target", 0.90)),
        max_fpr=cfg.get("max_fpr"),
    )
    return artifact, {
        "history": history,
        "pairs": int(len(pairs)),
        "best_epoch": best_epoch,
        "best_monitor": monitor,
        "best_score": best_score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FHIS v2 layerwise/ranking probe.")
    parser.add_argument("--config", default="data_generation/qwen25_fhis/configs/probe.yaml")
    parser.add_argument("--features", default=None)
    parser.add_argument("--metrics-output", default="classifier/v2_runs/layerwise_metrics.json")
    parser.add_argument("--probe-output", default="classifier/v2_runs/hidden_layerwise_probe_v2.joblib")
    args = parser.parse_args()

    config = load_config(args.config)
    features_path = args.features or config["paths"]["hidden_states"]
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

    config.setdefault(
        "probe_v2",
        {
            "layer_embed_dim": 256,
            "use_scalars": False,
            "mlp_dims": [256, 64],
            "dropout": 0.15,
            "lr": 3e-4,
            "weight_decay": 1e-4,
            "batch_size": 256,
            "max_epochs": 80,
            "patience": 12,
            "positive_weight_multiplier": 1.0,
            "ranking_weight": 0.5,
            "ranking_margin": 1.0,
            "ranking_pairs_per_epoch": 4096,
            "positive_recall_target": 0.90,
            "max_fpr": None,
            "monitor": "auprc",
        },
    )

    probe, train_info = train_layerwise_probe(train_rows, val_rows, config)
    test_scores = probe.predict_scores(stack_features(test_rows), rows=test_rows)
    y_test = labels(test_rows)
    metrics = evaluate_scores(test_rows, y_test, test_scores)
    metrics.update(
        {
            "decision_threshold": float(probe.decision_threshold),
            "positive_steps": int(y_test.sum()),
            "negative_steps": int(len(y_test) - y_test.sum()),
            "_split": {
                "train_steps": len(train_rows),
                "val_steps": len(val_rows),
                "test_steps": len(test_rows),
                "train_problems": len(splits["train"]),
                "val_problems": len(splits["val"]),
                "test_problems": len(splits["test"]),
            },
            "_train_info": train_info,
        }
    )

    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    probe_path = Path(args.probe_output)
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": probe,
            "probe_kind": "layerwise_mlp_v2",
            "feature_layers": probe.layer_ids,
            "feature_dim": int(stack_features(train_rows[:1]).shape[1]) if train_rows else 0,
            "trained_on": metrics["_split"],
            "config": config,
        },
        probe_path,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote probe to {probe_path}")


if __name__ == "__main__":
    main()
