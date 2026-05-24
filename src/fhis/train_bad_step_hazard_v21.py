from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from fhis.config import load_config
from fhis.metrics import evaluate_scores
from fhis.train_bad_step_detector_v21 import (
    apply_weight_overrides,
    inner_train_val,
    load_jsonl,
    rows_from_manifest,
    score_report,
    split_rows,
    write_score_details,
)
from fhis.train_probe import labels, stack_features
from fhis.train_probe_v2 import mean_scale, scalar_features, split_feature_tensor, standardize


def row_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row.get("sample_weight", 1.0)) for row in rows], dtype=np.float32)


def build_trace_indices(rows: list[dict[str, Any]]) -> list[list[int]]:
    by_trace: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_trace[str(row["trace_id"])].append(idx)
    traces = []
    for indices in by_trace.values():
        traces.append(sorted(indices, key=lambda i: int(rows[i]["step_index"])))
    traces.sort(key=lambda idxs: (str(rows[idxs[0]]["problem_id"]), str(rows[idxs[0]]["trace_id"])))
    return traces


class TraceDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        x_layers: np.ndarray,
        scalars: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        self.rows = rows
        self.x_layers = x_layers.astype(np.float32)
        self.scalars = scalars.astype(np.float32)
        self.y = y.astype(np.float32)
        self.weights = weights.astype(np.float32)
        self.traces = build_trace_indices(rows)

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        indices = np.asarray(self.traces[idx], dtype=np.int64)
        return {
            "indices": torch.from_numpy(indices),
            "x_layers": torch.from_numpy(self.x_layers[indices]),
            "scalars": torch.from_numpy(self.scalars[indices]),
            "y": torch.from_numpy(self.y[indices]),
            "weights": torch.from_numpy(self.weights[indices]),
        }


def collate_traces(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    lengths = torch.tensor([item["y"].shape[0] for item in batch], dtype=torch.long)
    return {
        "indices": pad_sequence([item["indices"] for item in batch], batch_first=True, padding_value=-1),
        "x_layers": pad_sequence([item["x_layers"] for item in batch], batch_first=True),
        "scalars": pad_sequence([item["scalars"] for item in batch], batch_first=True),
        "y": pad_sequence([item["y"] for item in batch], batch_first=True),
        "weights": pad_sequence([item["weights"] for item in batch], batch_first=True),
        "mask": torch.arange(int(lengths.max())).unsqueeze(0) < lengths.unsqueeze(1),
    }


class HazardProbeNet(nn.Module):
    def __init__(
        self,
        n_layers: int,
        hidden_size: int,
        scalar_dim: int,
        layer_embed_dim: int,
        sequence_hidden_dim: int,
        step_mlp_dim: int,
        sequence_num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layer_proj = nn.Sequential(
            nn.Linear(hidden_size, layer_embed_dim),
            nn.LayerNorm(layer_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layer_score = nn.Linear(layer_embed_dim, 1)
        self.step_proj = nn.Sequential(
            nn.Linear(layer_embed_dim + scalar_dim, step_mlp_dim),
            nn.LayerNorm(step_mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            step_mlp_dim,
            sequence_hidden_dim,
            num_layers=sequence_num_layers,
            batch_first=True,
            dropout=dropout if sequence_num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(sequence_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(sequence_hidden_dim, 1),
        )
        self.n_layers = int(n_layers)

    def forward(self, x_layers: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        batch, steps, layers, hidden = x_layers.shape
        x = x_layers.reshape(batch * steps, layers, hidden)
        emb = self.layer_proj(x)
        weights = torch.softmax(self.layer_score(emb).squeeze(-1), dim=-1)
        pooled = (emb * weights.unsqueeze(-1)).sum(dim=1).reshape(batch, steps, -1)
        step_features = self.step_proj(torch.cat([pooled, scalars], dim=-1))
        seq_out, _ = self.gru(step_features)
        return self.head(seq_out).squeeze(-1)


@dataclass
class HazardProbeArtifact:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    scalar_mean: np.ndarray
    scalar_scale: np.ndarray
    state_dict: dict[str, torch.Tensor]
    layer_ids: list[int]
    hidden_size: int
    layer_embed_dim: int
    sequence_hidden_dim: int
    step_mlp_dim: int
    sequence_num_layers: int
    dropout: float

    def build_model(self) -> HazardProbeNet:
        model = HazardProbeNet(
            n_layers=len(self.layer_ids),
            hidden_size=self.hidden_size,
            scalar_dim=int(self.scalar_mean.shape[0]),
            layer_embed_dim=self.layer_embed_dim,
            sequence_hidden_dim=self.sequence_hidden_dim,
            step_mlp_dim=self.step_mlp_dim,
            sequence_num_layers=self.sequence_num_layers,
            dropout=self.dropout,
        )
        model.load_state_dict(self.state_dict)
        model.eval()
        return model


def prepare_arrays(
    rows: list[dict[str, Any]],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    scalar_mean: np.ndarray,
    scalar_scale: np.ndarray,
    use_scalars: bool,
    n_layers: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_flat = standardize(stack_features(rows).astype(np.float32), feature_mean, feature_scale)
    x_layers, _ = split_feature_tensor(x_flat, n_layers)
    scalars = standardize(scalar_features(rows, use_scalars=use_scalars), scalar_mean, scalar_scale)
    return x_layers, scalars


def run_epoch(
    model: HazardProbeNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float,
    cfg: dict[str, Any],
) -> float:
    model.train()
    total_loss = 0.0
    total_weight = 0.0
    for batch in loader:
        x_layers = batch["x_layers"].to(device)
        scalars = batch["scalars"].to(device)
        y = batch["y"].to(device)
        weights = batch["weights"].to(device)
        mask = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_layers, scalars)
        loss = loss_fn(logits, y)
        effective = weights * mask.float()
        loss = (loss * effective).sum() / effective.sum().clamp_min(1.0)
        loss = loss + trace_level_regularizer(logits, y, mask, cfg)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * float(effective.sum().detach().cpu())
        total_weight += float(effective.sum().detach().cpu())
    return total_loss / max(total_weight, 1.0)


def trace_level_regularizer(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    cfg: dict[str, Any],
) -> torch.Tensor:
    correct_weight = float(cfg.get("correct_trace_max_logit_weight", 0.0))
    prefhis_weight = float(cfg.get("prefhis_trace_max_logit_weight", 0.0))
    positive_weight = float(cfg.get("positive_trace_logit_weight", 0.0))
    if correct_weight <= 0.0 and prefhis_weight <= 0.0 and positive_weight <= 0.0:
        return logits.new_zeros(())

    neg_margin = float(cfg.get("trace_negative_logit_margin", -1.0))
    pos_margin = float(cfg.get("trace_positive_logit_margin", 1.0))
    valid = mask.bool()
    positive = (y > 0.5) & valid
    has_positive = positive.any(dim=1)
    has_valid = valid.any(dim=1)
    total = logits.new_zeros(())

    if correct_weight > 0.0:
        correct_trace = has_valid & ~has_positive
        if correct_trace.any():
            correct_logits = logits[correct_trace].masked_fill(~valid[correct_trace], -1.0e9)
            max_correct = correct_logits.max(dim=1).values
            total = total + correct_weight * torch.nn.functional.softplus(max_correct - neg_margin).mean()

    if prefhis_weight > 0.0:
        pos_trace = has_positive
        if pos_trace.any():
            positions = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
            first_pos = torch.argmax(positive.to(torch.int64), dim=1, keepdim=True)
            prefhis_mask = valid & (positions < first_pos) & pos_trace.unsqueeze(1)
            has_prefhis = prefhis_mask.any(dim=1)
            if has_prefhis.any():
                prefhis_logits = logits[has_prefhis].masked_fill(~prefhis_mask[has_prefhis], -1.0e9)
                max_prefhis = prefhis_logits.max(dim=1).values
                total = total + prefhis_weight * torch.nn.functional.softplus(max_prefhis - neg_margin).mean()

    if positive_weight > 0.0:
        pos_trace = has_positive
        if pos_trace.any():
            positive_logits = logits[pos_trace].masked_fill(~positive[pos_trace], -1.0e9)
            max_positive = positive_logits.max(dim=1).values
            total = total + positive_weight * torch.nn.functional.softplus(pos_margin - max_positive).mean()

    return total


def predict_logits(
    model: HazardProbeNet,
    dataset: TraceDataset,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_traces)
    logits_out = np.zeros(len(dataset.rows), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x_layers"].to(device), batch["scalars"].to(device)).cpu().numpy()
            indices = batch["indices"].numpy()
            mask = batch["mask"].numpy()
            for b in range(indices.shape[0]):
                valid = mask[b]
                logits_out[indices[b, valid]] = logits[b, valid]
    return logits_out


def build_dataset(
    rows: list[dict[str, Any]],
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    scalar_mean: np.ndarray,
    scalar_scale: np.ndarray,
    use_scalars: bool,
    n_layers: int,
) -> TraceDataset:
    x_layers, scalars = prepare_arrays(
        rows,
        feature_mean,
        feature_scale,
        scalar_mean,
        scalar_scale,
        use_scalars,
        n_layers,
    )
    return TraceDataset(rows, x_layers, scalars, labels(rows), row_weights(rows))


def fit_hazard_probe(
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[HazardProbeArtifact, dict[str, Any]]:
    cfg = config.setdefault("bad_step_hazard_v21", {})
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    use_scalars = bool(cfg.get("use_scalars", True))
    layer_ids = [int(x) for x in train_rows[0]["layer_ids"]]
    n_layers = len(layer_ids)
    x_train_flat = stack_features(train_rows).astype(np.float32)
    feature_mean, feature_scale = mean_scale(x_train_flat)
    _, hidden_size = split_feature_tensor(x_train_flat, n_layers)

    s_train_raw = scalar_features(train_rows, use_scalars=use_scalars)
    scalar_mean, scalar_scale = mean_scale(s_train_raw)

    train_dataset = build_dataset(
        train_rows,
        feature_mean,
        feature_scale,
        scalar_mean,
        scalar_scale,
        use_scalars,
        n_layers,
    )
    val_dataset = build_dataset(
        val_rows,
        feature_mean,
        feature_scale,
        scalar_mean,
        scalar_scale,
        use_scalars,
        n_layers,
    )

    device = torch.device(str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    model = HazardProbeNet(
        n_layers=n_layers,
        hidden_size=hidden_size,
        scalar_dim=int(scalar_mean.shape[0]),
        layer_embed_dim=int(cfg.get("layer_embed_dim", 192)),
        sequence_hidden_dim=int(cfg.get("sequence_hidden_dim", 192)),
        step_mlp_dim=int(cfg.get("step_mlp_dim", 192)),
        sequence_num_layers=int(cfg.get("sequence_num_layers", 1)),
        dropout=float(cfg.get("dropout", 0.20)),
    ).to(device)

    y_train = labels(train_rows).astype(np.float32)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor(
        [(neg / max(pos, 1.0)) * float(cfg.get("positive_weight_multiplier", 0.75))],
        dtype=torch.float32,
        device=device,
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 2e-4)),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.get("trace_batch_size", 32)),
        shuffle=True,
        collate_fn=collate_traces,
        generator=torch.Generator().manual_seed(seed),
    )

    max_epochs = int(cfg.get("max_epochs", 90))
    patience = int(cfg.get("patience", 14))
    eval_batch_size = int(cfg.get("eval_trace_batch_size", 64))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    y_val = labels(val_rows)

    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, loader, optimizer, loss_fn, device, grad_clip, cfg)
        val_logits = predict_logits(model, val_dataset, device, eval_batch_size)
        val_scores = 1.0 / (1.0 + np.exp(-val_logits))
        if len(np.unique(y_val)) == 2:
            val_auprc = float(average_precision_score(y_val, val_scores))
            val_auroc = float(roc_auc_score(y_val, val_scores))
        else:
            val_auprc = float("nan")
            val_auroc = float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auprc": val_auprc, "val_auroc": val_auroc})
        if np.isfinite(val_auprc) and val_auprc > best_score + float(cfg.get("min_delta", 1e-4)):
            best_score = val_auprc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    artifact = HazardProbeArtifact(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        scalar_mean=scalar_mean,
        scalar_scale=scalar_scale,
        state_dict=best_state,
        layer_ids=layer_ids,
        hidden_size=int(hidden_size),
        layer_embed_dim=int(cfg.get("layer_embed_dim", 192)),
        sequence_hidden_dim=int(cfg.get("sequence_hidden_dim", 192)),
        step_mlp_dim=int(cfg.get("step_mlp_dim", 192)),
        sequence_num_layers=int(cfg.get("sequence_num_layers", 1)),
        dropout=float(cfg.get("dropout", 0.20)),
    )
    return artifact, {
        "best_epoch": best_epoch,
        "best_val_auprc": best_score,
        "history": history,
        "num_train_traces": len(train_dataset),
        "num_val_traces": len(val_dataset),
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_calibrator(logits: np.ndarray, y: np.ndarray, method: str) -> Any:
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(logits, y)
        return calibrator
    if method == "platt":
        calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        calibrator.fit(logits.reshape(-1, 1), y)
        return calibrator
    raise ValueError(f"unknown calibration method: {method}")


def calibrator_predict(calibrator: Any, logits: np.ndarray, method: str) -> np.ndarray:
    if method == "isotonic":
        return calibrator.predict(logits)
    if method == "platt":
        return calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
    raise ValueError(f"unknown calibration method: {method}")



def budget_threshold_metrics(
    rows: list[dict[str, Any]],
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
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
    wrong_with_fhis = 0
    wrong_detected = 0
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
            wrong_with_fhis += 1
            fhis_step = min(int(row["step_index"]) for row, _ in pairs if int(row["label"]) == 1)
            wrong_detected += int(any(flag and int(row["label"]) == 1 for row, flag in pairs))
            prefhis_stopped += int(any(flag and int(row["step_index"]) < fhis_step for row, flag in pairs))
        if triggered:
            first_trigger_total += 1
            first_trigger_positive += int(int(triggered[0][0]["label"]) == 1)

    first_trigger_fp = first_trigger_total - first_trigger_positive
    return {
        "threshold": float(threshold),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "fp_per_tp": fp / max(tp, 1),
        "correct_trace_false_stop_rate": correct_stopped / max(correct, 1),
        "pre_fhis_false_stop_rate": prefhis_stopped / max(wrong_with_fhis, 1),
        "wrong_trace_recall": wrong_detected / max(wrong_with_fhis, 1),
        "first_trigger_precision": first_trigger_positive / max(first_trigger_total, 1),
        "first_trigger_fp_per_tp": first_trigger_fp / max(first_trigger_positive, 1),
        "triggered_traces": float(first_trigger_total),
    }


def recall_budget_thresholds(
    rows: list[dict[str, Any]],
    y: np.ndarray,
    scores: np.ndarray,
    budgets: list[float],
) -> list[dict[str, float]]:
    candidates = np.unique(scores)
    if len(candidates) > 4000:
        candidates = np.quantile(scores, np.linspace(0.0, 1.0, 4000))
    table: list[dict[str, float]] = []
    for budget in budgets:
        feasible_step: list[dict[str, float]] = []
        feasible_first: list[dict[str, float]] = []
        for tau in candidates:
            item = budget_threshold_metrics(rows, y, scores, float(tau))
            if item["tp"] <= 0:
                continue
            if item["fp_per_tp"] <= budget:
                feasible_step.append(item)
            if item["first_trigger_fp_per_tp"] <= budget:
                feasible_first.append(item)
        for mode, feasible in [("step", feasible_step), ("first_trigger", feasible_first)]:
            if feasible:
                best = max(
                    feasible,
                    key=lambda x: (
                        x["recall"],
                        x["wrong_trace_recall"],
                        -x["correct_trace_false_stop_rate"],
                        -x["fp_per_tp"],
                    ),
                )
            else:
                best = budget_threshold_metrics(rows, y, scores, float(np.max(scores)))
            best = dict(best)
            best["fp_per_tp_budget"] = float(budget)
            best["budget_mode"] = mode
            table.append(best)
    return table

def main() -> None:
    parser = argparse.ArgumentParser(description="Train causal hazard bad-step detector v2.1.")
    parser.add_argument("--config", default="data_generation/qwen25_fhis/configs/probe.yaml")
    parser.add_argument("--features", default="data_generation/qwen25_fhis/features/step_hidden_states_codex_clean.pt")
    parser.add_argument("--manifest", default="classifier/v2_runs/bad_step_v21/manifest.jsonl")
    parser.add_argument("--output-dir", default="classifier/v2_runs/bad_step_v21/hazard_causal_gru")
    parser.add_argument("--positive-weight-multiplier", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--layer-embed-dim", type=int, default=None)
    parser.add_argument("--sequence-hidden-dim", type=int, default=None)
    parser.add_argument("--step-mlp-dim", type=int, default=None)
    parser.add_argument("--sequence-num-layers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--trace-batch-size", type=int, default=None)
    parser.add_argument("--positive-sample-weight", type=float, default=None)
    parser.add_argument("--hard-positive-weight", type=float, default=None)
    parser.add_argument("--hard-negative-weight", type=float, default=None)
    parser.add_argument("--correct-negative-weight", type=float, default=None)
    parser.add_argument("--prefhis-negative-weight", type=float, default=None)
    parser.add_argument("--calibration-method", choices=["isotonic", "platt"], default="isotonic")
    parser.add_argument("--correct-trace-max-logit-weight", type=float, default=None)
    parser.add_argument("--prefhis-trace-max-logit-weight", type=float, default=None)
    parser.add_argument("--positive-trace-logit-weight", type=float, default=None)
    parser.add_argument("--trace-negative-logit-margin", type=float, default=None)
    parser.add_argument("--trace-positive-logit-margin", type=float, default=None)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    cfg = config.setdefault("bad_step_hazard_v21", {})
    cfg.setdefault("use_scalars", True)
    cfg.setdefault("layer_embed_dim", 192)
    cfg.setdefault("sequence_hidden_dim", 192)
    cfg.setdefault("step_mlp_dim", 192)
    cfg.setdefault("sequence_num_layers", 1)
    cfg.setdefault("dropout", 0.20)
    cfg.setdefault("lr", 2e-4)
    cfg.setdefault("weight_decay", 2e-4)
    cfg.setdefault("trace_batch_size", 32)
    cfg.setdefault("max_epochs", 90)
    cfg.setdefault("patience", 14)
    for key, value in {
        "positive_weight_multiplier": args.positive_weight_multiplier,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "layer_embed_dim": args.layer_embed_dim,
        "sequence_hidden_dim": args.sequence_hidden_dim,
        "step_mlp_dim": args.step_mlp_dim,
        "sequence_num_layers": args.sequence_num_layers,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "trace_batch_size": args.trace_batch_size,
        "correct_trace_max_logit_weight": args.correct_trace_max_logit_weight,
        "prefhis_trace_max_logit_weight": args.prefhis_trace_max_logit_weight,
        "positive_trace_logit_weight": args.positive_trace_logit_weight,
        "trace_negative_logit_margin": args.trace_negative_logit_margin,
        "trace_positive_logit_margin": args.trace_positive_logit_margin,
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

    artifact, train_info = fit_hazard_probe(train_rows, val_rows, config)
    model = artifact.build_model()
    device = torch.device(str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    model.to(device)
    eval_batch_size = int(cfg.get("eval_trace_batch_size", 64))

    def make_dataset(split: list[dict[str, Any]]) -> TraceDataset:
        return build_dataset(
            split,
            artifact.feature_mean,
            artifact.feature_scale,
            artifact.scalar_mean,
            artifact.scalar_scale,
            bool(cfg.get("use_scalars", True)),
            len(artifact.layer_ids),
        )

    calibration_dataset = make_dataset(calibration_rows)
    cal_logits = predict_logits(model, calibration_dataset, device, eval_batch_size)
    y_cal = labels(calibration_rows)
    calibrator = fit_calibrator(cal_logits, y_cal, args.calibration_method)
    cal_scores = calibrator_predict(calibrator, cal_logits, args.calibration_method)

    reports: dict[str, Any] = {
        "split_sizes": {
            "train_inner": len(train_rows),
            "val_inner": len(val_rows),
            "calibration": len(calibration_rows),
            "hard_dev": len(hard_dev_rows),
        },
        "train_info": train_info,
        "raw_sigmoid": {},
        "calibrated": {},
        "effective_config": cfg,
        "calibration_method": args.calibration_method,
        "weight_overrides": {
            "positive_sample_weight": args.positive_sample_weight,
            "hard_positive_weight": args.hard_positive_weight,
            "hard_negative_weight": args.hard_negative_weight,
            "correct_negative_weight": args.correct_negative_weight,
            "prefhis_negative_weight": args.prefhis_negative_weight,
        },
    }

    for name, split in [("val_inner", val_rows), ("calibration", calibration_rows), ("hard_dev", hard_dev_rows)]:
        dataset = make_dataset(split)
        logits = predict_logits(model, dataset, device, eval_batch_size)
        raw_scores = sigmoid(logits)
        calibrated = calibrator_predict(calibrator, logits, args.calibration_method)
        reports["raw_sigmoid"][name] = score_report(name, split, raw_scores)
        reports["raw_sigmoid"][name]["recall_budget_thresholds"] = recall_budget_thresholds(
            split, labels(split), raw_scores, [1.0, 1.5, 2.0, 3.0]
        )
        reports["calibrated"][name] = score_report(name, split, calibrated)
        reports["calibrated"][name]["recall_budget_thresholds"] = recall_budget_thresholds(
            split, labels(split), calibrated, [1.0, 1.5, 2.0, 3.0]
        )
        write_score_details(out / f"{name}_score_details.jsonl", split, raw_scores, calibrated)

    frac_pos, mean_pred = calibration_curve(y_cal, cal_scores, n_bins=10, strategy="quantile")
    reports["calibration_curve"] = {
        "fraction_of_positives": [float(x) for x in frac_pos],
        "mean_predicted_value": [float(x) for x in mean_pred],
    }

    (out / "calibration_report.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2))
    joblib.dump(
        {
            "model": artifact,
            "calibrator": calibrator,
            "probe_kind": f"bad_step_detector_v21_causal_hazard_gru_{args.calibration_method}",
            "config": config,
            "train_info": train_info,
            "calibration_report_path": str(out / "calibration_report.json"),
        },
        out / "probe_calibrated.joblib",
    )
    print(json.dumps(reports["split_sizes"], ensure_ascii=False, indent=2))
    print(json.dumps(reports["calibrated"]["hard_dev"]["precision_thresholds"], ensure_ascii=False, indent=2))
    print(json.dumps(reports["calibrated"]["hard_dev"]["recall_budget_thresholds"], ensure_ascii=False, indent=2))
    print(f"Wrote {out / 'probe_calibrated.joblib'}")
    print(f"Wrote {out / 'calibration_report.json'}")


if __name__ == "__main__":
    main()
