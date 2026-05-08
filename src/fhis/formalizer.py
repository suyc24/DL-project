from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class FormalizationRequest:
    problem: str
    previous_steps: list[str]
    current_step_index: int
    current_step: str


class StepFormalizer(Protocol):
    def formalize(self, request: FormalizationRequest) -> str:
        ...


def strip_code_fence(text: str) -> str:
    text = text.strip()
    fence_matches = list(re.finditer(r"```(?:lean4?|Lean4?)?\s*(.*?)```", text, flags=re.S))
    if fence_matches:
        return fence_matches[-1].group(1).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    code_start = re.search(r"(?m)^(?:import\s+|theorem\s+|lemma\s+|example\s+)", text)
    if code_start:
        return text[code_start.start() :].strip()
    return text


def build_formalizer_prompt(request: FormalizationRequest) -> str:
    previous = "\n".join(
        f"Step {i + 1}: {text}" for i, text in enumerate(request.previous_steps)
    )
    if not previous:
        previous = "(none)"
    header = "import Mathlib\n\nopen Real\n"
    informal_problem = f"""Original problem:
{request.problem}

Previously accepted natural-language steps:
{previous}

Current natural-language step to verify:
Step {request.current_step_index}: {request.current_step}

Formalize the claim that this current step is valid under the original problem
and the previously accepted steps. The Lean code must contain a complete proof.
Do not use `sorry`, `admit`, `axiom`, or unfinished placeholders.
If the current step cannot be faithfully formalized with a complete proof, output:

-- formalization_failed
"""
    return (
        "Please autoformalize the following problem in Lean 4 with a header. "
        "Use the following theorem names: current_step_valid.\n\n"
        f"{informal_problem}\n"
        "Your code should start with:\n"
        "```Lean4\n"
        f"{header}"
        "```\n"
    )


class NullFormalizer:
    def formalize(self, request: FormalizationRequest) -> str:
        return "-- formalization_failed"


class TransformersFormalizer:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        patch_transformers_safetensors_metadata()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        torch_dtype = "auto"
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32
        device_map = device if device == "auto" or device.startswith("cuda") else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            device_map=device_map,
        )
        if device == "cpu":
            self.model.to("cpu")
        self.model.eval()
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)

    def formalize(self, request: FormalizationRequest) -> str:
        import torch

        prompt = build_formalizer_prompt(request)
        messages = [
            {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
            {"role": "user", "content": prompt},
        ]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            text += "<think>"
        except TypeError:
            text = prompt
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        do_sample = self.temperature > 0
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, encoded["input_ids"].shape[1] :]
        return strip_code_fence(self.tokenizer.decode(generated, skip_special_tokens=True))


def patch_transformers_safetensors_metadata() -> None:
    """Allow loading safetensors shards that omit optional metadata.

    Some HF model shards, including StepFun-Formalizer-7B at the time this was
    tested, have `safe_open(...).metadata() is None`. Transformers 4.47 assumes
    this is a dict and crashes before loading otherwise valid tensors.
    """
    import transformers.modeling_utils as modeling_utils
    from safetensors import safe_open
    from safetensors.torch import load_file as safe_load_file

    if getattr(modeling_utils, "_fhis_safetensors_metadata_patch", False):
        return
    original = modeling_utils.load_state_dict

    def load_state_dict_with_missing_metadata(
        checkpoint_file,
        is_quantized=False,
        map_location=None,
        weights_only=True,
    ):
        path = str(checkpoint_file)
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt") as f:
                metadata = f.metadata()
            if metadata is None:
                return safe_load_file(path)
        return original(
            checkpoint_file,
            is_quantized=is_quantized,
            map_location=map_location,
            weights_only=weights_only,
        )

    modeling_utils.load_state_dict = load_state_dict_with_missing_metadata
    modeling_utils._fhis_safetensors_metadata_patch = True


def build_formalizer(config: dict) -> StepFormalizer:
    formalizer_cfg = config.get("formalizer", {})
    backend_value = formalizer_cfg.get("backend", "null")
    backend = "null" if backend_value is None else str(backend_value).lower()
    if backend in {"null", "none"}:
        return NullFormalizer()
    if backend in {"transformers", "hf", "stepfun"}:
        return TransformersFormalizer(
            model_name=str(formalizer_cfg.get("model", "StepFun-AI/StepFun-Formalizer-7B")),
            device=str(formalizer_cfg.get("device", "auto")),
            dtype=str(formalizer_cfg.get("dtype", "auto")),
            max_new_tokens=int(formalizer_cfg.get("max_new_tokens", 1024)),
            temperature=float(formalizer_cfg.get("temperature", 0.0)),
            top_p=float(formalizer_cfg.get("top_p", 0.95)),
        )
    raise ValueError(f"Unsupported formalizer.backend={backend!r}")
