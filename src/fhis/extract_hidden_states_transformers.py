from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from fhis.config import load_config, selected_layers_for_model
from fhis.extract_hidden_states_vllm import build_feature_rows
from fhis.io import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug fallback hidden-state extractor using transformers, not for final vLLM runs."
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--traces", default=None)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    layer_ids = selected_layers_for_model(config)
    keep_confidence = config["labels"].get("keep_confidence", "high")
    traces_path = args.traces or config["paths"]["generated_traces"]
    labels_path = args.labels or config["paths"]["labels"]
    output_path = Path(args.output or config["paths"]["hidden_states"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = {row["trace_id"]: row for row in read_jsonl(labels_path)}
    traces = [row for row in read_jsonl(traces_path) if row["trace_id"] in labels]
    if args.limit is not None:
        traces = traces[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
        device_map=args.device if args.device.startswith("cuda") else None,
    )
    if args.device == "cpu":
        model.to("cpu")
    model.eval()

    feature_rows = []
    with torch.no_grad():
        for trace in tqdm(traces, desc="extracting states"):
            text = trace["prompt"] + trace["completion"]
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            # HF returns embeddings at index 0, then post-block states at index layer+1.
            selected = torch.stack(
                [outputs.hidden_states[layer_id + 1][0] for layer_id in layer_ids],
                dim=1,
            )
            feature_rows.extend(
                build_feature_rows(
                    trace,
                    labels[trace["trace_id"]],
                    selected.cpu(),
                    tokenizer,
                    layer_ids,
                    keep_confidence,
                )
            )

    torch.save({"rows": feature_rows, "layer_ids": layer_ids, "model": model_cfg["name"]}, output_path)
    print(f"Wrote {len(feature_rows)} labeled step features to {output_path}")


if __name__ == "__main__":
    main()
