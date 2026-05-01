from __future__ import annotations

import argparse
from typing import Any

from tqdm import tqdm

from fhis.config import load_config
from fhis.io import append_jsonl, read_existing_ids, read_jsonl
from fhis.labeling import build_label_prompt, normalize_label, parse_json_object


def call_openai_label(client: Any, model: str, prompt: str) -> dict[str, Any]:
    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=0,
        )
        text = response.output_text
    except Exception:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = response.choices[0].message.content
    return parse_json_object(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label generated traces with a GPT annotator.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--traces", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from openai import OpenAI

    config = load_config(args.config)
    model = config["labels"]["annotator_model"]
    traces_path = args.traces or config["paths"]["generated_traces"]
    output_path = args.output or config["paths"]["labels"]
    traces = list(read_jsonl(traces_path))
    if args.limit is not None:
        traces = traces[: args.limit]
    if args.resume:
        done = read_existing_ids(output_path, "trace_id")
        traces = [t for t in traces if str(t["trace_id"]) not in done]

    client = OpenAI()
    for trace in tqdm(traces, desc="labeling traces"):
        prompt = build_label_prompt(trace)
        raw = call_openai_label(client, model, prompt)
        label = normalize_label(raw, trace_id=str(trace["trace_id"]))
        append_jsonl(output_path, [label])

    print(f"Wrote labels for {len(traces)} traces to {output_path}")


if __name__ == "__main__":
    main()
