import json
import os
import random
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "external/coconut")
from dataset import get_dataset, get_question_latent_dataset

from src.data import extract_answer_after_marker, normalize_answer
from src.coconut_utils import build_base_lm, wrap_in_coconut
from src.steering import SteeredCoconut


def _load_cfg(path):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _acc_text_model(model, tokenizer, rows, prompt_style, max_new_tokens):
    model.eval()
    ok = 0
    for ex in rows:
        if prompt_style == "no_cot":
            prompt = f"{ex['question']}\n### "
        else:
            prompt = f"{ex['question']}\nLet's think step by step.\n### "
        toks = tokenizer(prompt, return_tensors="pt").to(_dev())
        with torch.no_grad():
            out = model.generate(
                **toks,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        pred = extract_answer_after_marker(tokenizer.decode(out[0], skip_special_tokens=True))
        ok += int(normalize_answer(pred) == normalize_answer(ex["answer"]))
    return ok / max(1, len(rows))


def _acc_coconut(model, tokenizer, q_ds, max_new_tokens):
    model.eval()
    ok = 0
    for item in q_ds:
        with torch.no_grad():
            out = model.generate(
                input_ids=item["input_ids"].unsqueeze(0).to(_dev()),
                attention_mask=item["attention_mask"].unsqueeze(0).to(_dev()),
                max_new_tokens=max_new_tokens,
                synced_gpus=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        pred = extract_answer_after_marker(tokenizer.decode(out[0], skip_special_tokens=True))
        ok += int(normalize_answer(pred) == normalize_answer(item["answer"]))
    return ok / max(1, len(q_ds))


def run(cfg):
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    test_rows = json.load(open(cfg["paths"]["test_path"], "r", encoding="utf-8"))
    max_new_tokens = cfg["phase4"]["max_new_tokens"]

    base, tok = build_base_lm(cfg["model_id"], add_latent_tokens=False)
    no_cot_acc = _acc_text_model(base, tok, test_rows, "no_cot", max_new_tokens)
    text_cot_acc = _acc_text_model(base, tok, test_rows, "text_cot", max_new_tokens)

    ckpt = json.load(open(cfg["paths"]["coconut_ckpt_meta"], "r", encoding="utf-8"))["ckpt_path"]
    c_base, c_tok = build_base_lm(
        cfg["model_id"],
        add_latent_tokens=True,
        load_model_path=ckpt,
    )
    coconut = wrap_in_coconut(c_base, c_tok).to(_dev())
    cobj = SimpleNamespace(**{**cfg["curriculum"], **cfg})
    q_ds = get_question_latent_dataset(
        cfg["curriculum"]["max_latent_stage"],
        get_dataset(cfg["paths"]["test_path"], c_tok),
        cobj,
        c_tok.convert_tokens_to_ids("<|start-latent|>"),
        c_tok.convert_tokens_to_ids("<|latent|>"),
        c_tok.convert_tokens_to_ids("<|end-latent|>"),
    )
    coconut_alpha0_acc = _acc_coconut(coconut, c_tok, q_ds, max_new_tokens)

    alpha = json.load(open(os.path.join(cfg["paths"]["alpha_dir"], "best_alpha.json"), "r", encoding="utf-8"))["alpha"]

    def make_alpha_fn(a):
        return lambda: torch.tensor(a, device=_dev(), dtype=torch.float32)

    d_model = c_base.get_input_embeddings().weight.shape[1]
    random.seed(cfg["phase4"]["random_seed"])
    r = torch.randn(d_model, device=_dev())
    r = r / (r.norm() + 1e-12)
    steer_noise = SteeredCoconut(
        base_causallm=c_base,
        latent_token_id=c_tok.convert_tokens_to_ids("<|latent|>"),
        start_latent_id=c_tok.convert_tokens_to_ids("<|start-latent|>"),
        end_latent_id=c_tok.convert_tokens_to_ids("<|end-latent|>"),
        eos_token_id=c_tok.eos_token_id,
        alpha_fn=make_alpha_fn(alpha),
        v_truth=r,
        gamma=cfg["phase3"]["gamma"],
    ).to(_dev())
    coconut_random_noise_acc = _acc_coconut(steer_noise, c_tok, q_ds, max_new_tokens)

    v_truth = torch.load(os.path.join(cfg["paths"]["vectors_dir"], "source_a_dom_shared.pt"), map_location=_dev())
    v_truth = v_truth / (v_truth.norm() + 1e-12)
    steer_truth = SteeredCoconut(
        base_causallm=c_base,
        latent_token_id=c_tok.convert_tokens_to_ids("<|latent|>"),
        start_latent_id=c_tok.convert_tokens_to_ids("<|start-latent|>"),
        end_latent_id=c_tok.convert_tokens_to_ids("<|end-latent|>"),
        eos_token_id=c_tok.eos_token_id,
        alpha_fn=make_alpha_fn(alpha),
        v_truth=v_truth,
        gamma=cfg["phase3"]["gamma"],
    ).to(_dev())
    coconut_truth_alpha_star_acc = _acc_coconut(steer_truth, c_tok, q_ds, max_new_tokens)

    summary = {
        "model_id": cfg["model_id"],
        "conditions": {
            "no_cot": no_cot_acc,
            "text_cot": text_cot_acc,
            "coconut_alpha0": coconut_alpha0_acc,
            "coconut_random_noise": coconut_random_noise_acc,
            "coconut_truth_alpha_star": coconut_truth_alpha_star_acc,
        },
    }
    with open(os.path.join(cfg["paths"]["results_dir"], "phase4_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    run(_load_cfg(sys.argv[1]))
