from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create basic v0 figures from probe outputs.")
    parser.add_argument("--metrics", default="results/probe_metrics.json")
    parser.add_argument("--layer-sweep", default="results/layer_sweep.csv")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    metrics = {k: v for k, v in metrics.items() if not k.startswith("_")}

    names = list(metrics)
    recall1 = [metrics[name]["recall_at_1"] for name in names]
    recall2 = [metrics[name]["recall_at_2"] for name in names]
    coverage = [metrics[name]["top_30pct_budget_coverage"] for name in names]

    plt.figure(figsize=(10, 4))
    x = range(len(names))
    plt.bar([i - 0.18 for i in x], recall1, width=0.36, label="Recall@1")
    plt.bar([i + 0.18 for i in x], recall2, width=0.36, label="Recall@2")
    plt.xticks(list(x), names, rotation=35, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "recall_at_k.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(names, coverage)
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("FHIS caught by top 30% steps")
    plt.tight_layout()
    plt.savefig(figures_dir / "top_budget_coverage.png", dpi=200)
    plt.close()

    layer_path = Path(args.layer_sweep)
    if layer_path.exists() and layer_path.stat().st_size:
        df = pd.read_csv(layer_path)
        if not df.empty:
            plt.figure(figsize=(6, 4))
            plt.plot(df["layer"], df["auroc"], marker="o", label="AUROC")
            plt.plot(df["layer"], df["recall_at_1"], marker="o", label="Recall@1")
            plt.xlabel("Layer")
            plt.ylim(0, 1)
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / "layer_sweep.png", dpi=200)
            plt.close()

    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
