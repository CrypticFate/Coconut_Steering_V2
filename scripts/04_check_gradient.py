import json
import os
import sys
from types import SimpleNamespace

import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

sys.path.insert(0, "external/coconut")
from dataset import MyCollator, get_cot_latent_dataset, get_dataset

from src.coconut_utils import build_base_lm
from src.steering import SteeredCoconut


def dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
    ckpt = json.load(open(cfg["paths"]["coconut_ckpt_meta"], "r", encoding="utf-8"))["ckpt_path"]
    base_model, tokenizer = build_base_lm(
        cfg["model_id"],
        add_latent_tokens=True,
        load_model_path=ckpt,
    )
    v_truth = torch.load(os.path.join(cfg["paths"]["vectors_dir"], "source_a_dom_shared.pt"), map_location=dev())
    v_truth = v_truth / (v_truth.norm() + 1e-12)

    theta = torch.nn.Parameter(torch.tensor(-3.89, device=dev(), dtype=torch.float32))

    def alpha_fn():
        return cfg["phase3"]["alpha_max"] * torch.sigmoid(theta)

    model = SteeredCoconut(
        base_causallm=base_model,
        latent_token_id=tokenizer.convert_tokens_to_ids("<|latent|>"),
        start_latent_id=tokenizer.convert_tokens_to_ids("<|start-latent|>"),
        end_latent_id=tokenizer.convert_tokens_to_ids("<|end-latent|>"),
        eos_token_id=tokenizer.eos_token_id,
        alpha_fn=alpha_fn,
        v_truth=v_truth,
        gamma=cfg["phase3"]["gamma"],
    ).to(dev())
    for p in model.base_causallm.parameters():
        p.requires_grad = False

    base_ds = get_dataset(cfg["paths"]["val_path"], tokenizer)
    cobj = SimpleNamespace(**{**cfg["curriculum"], **cfg})
    ds = get_cot_latent_dataset(
        scheduled_stage=cfg["curriculum"]["max_latent_stage"],
        base_dataset=base_ds,
        configs=cobj,
        start_id=tokenizer.convert_tokens_to_ids("<|start-latent|>"),
        latent_id=tokenizer.convert_tokens_to_ids("<|latent|>"),
        end_id=tokenizer.convert_tokens_to_ids("<|end-latent|>"),
        no_special_marker=False,
    )
    collator = MyCollator(tokenizer, latent_id=tokenizer.convert_tokens_to_ids("<|latent|>"), label_pad_token_id=-100)
    ex = collator([ds[0]])
    ex = {k: v.to(dev()) if torch.is_tensor(v) else v for k, v in ex.items()}

    out = model(**ex)
    out.loss.backward()
    g_analytic = theta.grad.item()

    eps = 1e-3
    with torch.no_grad():
        theta.add_(eps)
        lp = model(**ex).loss.item()
        theta.sub_(2 * eps)
        lm = model(**ex).loss.item()
        theta.add_(eps)
    g_numeric = (lp - lm) / (2 * eps)
    rel_err = abs(g_analytic - g_numeric) / (abs(g_numeric) + 1e-6)
    print(f"analytic={g_analytic:.4e} numeric={g_numeric:.4e} rel_err={rel_err:.4f}")
    assert rel_err < 0.05, "GRADIENT FLOW BROKEN — fix before alpha tuning"


if __name__ == "__main__":
    main()
