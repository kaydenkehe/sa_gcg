"""Hugging Face model loading with sane defaults for attack-time use.

Single A6000 (48 GB) fits fp16 7B easily. 13B fp16 needs ~26 GB. 70B needs
multi-GPU; out of scope for the primary target but supported for transfer
eval (set ``device_map='auto'``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .chat_template import (
    LLAMA2_SPEC,
    LLAMA3_SPEC,
    MISTRAL_SPEC,
    VICUNA_SPEC,
    ChatSpec,
)


@dataclass
class LoadedModel:
    model: object  # transformers PreTrainedModel
    tokenizer: object  # transformers PreTrainedTokenizerBase
    spec: ChatSpec
    name_or_path: str
    dtype: str
    device: str


SPEC_BY_PREFIX = (
    ("meta-llama/Llama-2", LLAMA2_SPEC),
    ("NousResearch/Llama-2", LLAMA2_SPEC),
    ("meta-llama/Meta-Llama-3", LLAMA3_SPEC),
    ("meta-llama/Llama-3", LLAMA3_SPEC),
    ("lmsys/vicuna", VICUNA_SPEC),
    ("mistralai/Mistral", MISTRAL_SPEC),
)


def spec_for(name_or_path: str) -> ChatSpec:
    """Pick the chat spec for a model identifier."""
    for prefix, spec in SPEC_BY_PREFIX:
        if name_or_path.startswith(prefix) or prefix.lower() in name_or_path.lower():
            return spec
    if "llama-2" in name_or_path.lower():
        return LLAMA2_SPEC
    if "llama-3" in name_or_path.lower():
        return LLAMA3_SPEC
    if "vicuna" in name_or_path.lower():
        return VICUNA_SPEC
    if "mistral" in name_or_path.lower():
        return MISTRAL_SPEC
    raise ValueError(
        f"No chat spec known for {name_or_path!r}. Add it to SPEC_BY_PREFIX or "
        f"pass an explicit ChatSpec to the attack."
    )


def load_chat_model(
    name_or_path: str,
    *,
    dtype: str = "fp16",
    device: str = "cuda",
    device_map: Optional[str] = None,
    trust_remote_code: bool = False,
) -> LoadedModel:
    """Load a HF chat model + tokenizer with the right chat spec attached."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    th_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path, use_fast=True, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict = {
        "dtype": th_dtype,
        "trust_remote_code": trust_remote_code,
        "use_safetensors": True,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)
    if device_map is None:
        model = model.to(device)
    model.eval()
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        spec=spec_for(name_or_path),
        name_or_path=name_or_path,
        dtype=dtype,
        device=device,
    )
