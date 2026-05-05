"""
Official Coconut integration utilities (no LoRA/PEFT).
"""
import sys
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "external/coconut")
from coconut import Coconut  # noqa: E402

LATENT_TOKENS = ["<|start-latent|>", "<|latent|>", "<|end-latent|>"]


def _pick_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def build_base_lm(
    model_id: str,
    add_latent_tokens: bool,
    load_model_path: Optional[str] = None,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    model_source = load_model_path if load_model_path else model_id
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if add_latent_tokens:
        tokenizer.add_tokens(LATENT_TOKENS)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    common_kwargs = {
        "attn_implementation": "sdpa",
        "device_map": "auto" if torch.cuda.is_available() else None,
        "trust_remote_code": True,
    }
    # Compatibility across transformers versions:
    # newer accepts `dtype`, while many stable releases expect `torch_dtype`.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=_pick_dtype(),
            **common_kwargs,
        )
    except TypeError as e:
        if "unexpected keyword argument 'dtype'" not in str(e):
            raise
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=_pick_dtype(),
            **common_kwargs,
        )

    if add_latent_tokens:
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            ll_id = tokenizer.convert_tokens_to_ids("<<")
            if ll_id is None or ll_id == tokenizer.unk_token_id:
                ll_id = tokenizer.convert_tokens_to_ids("<")
            if ll_id is None or ll_id == tokenizer.unk_token_id:
                ll_id = tokenizer.eos_token_id
            init_embed = model.get_input_embeddings().weight[ll_id].clone()
            for tok in LATENT_TOKENS:
                tid = tokenizer.convert_tokens_to_ids(tok)
                model.get_input_embeddings().weight[tid] = init_embed.clone()
                if model.get_output_embeddings() is not None:
                    model.get_output_embeddings().weight[tid] = init_embed.clone()

    return model, tokenizer


def wrap_in_coconut(base_model, tokenizer):
    return Coconut(
        base_causallm=base_model,
        latent_token_id=tokenizer.convert_tokens_to_ids("<|latent|>"),
        start_latent_id=tokenizer.convert_tokens_to_ids("<|start-latent|>"),
        end_latent_id=tokenizer.convert_tokens_to_ids("<|end-latent|>"),
        eos_token_id=tokenizer.eos_token_id,
    )
