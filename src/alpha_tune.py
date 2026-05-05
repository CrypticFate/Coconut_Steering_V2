import json
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "external/coconut")
from dataset import MyCollator, get_cot_latent_dataset, get_dataset, get_question_latent_dataset

from src.data import extract_answer_after_marker, normalize_answer
from src.coconut_utils import build_base_lm
from src.steering import SteeredCoconut


def _load_cfg(path):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _eval_acc(model, tokenizer, q_ds, max_new_tokens):
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
    os.makedirs(cfg["paths"]["alpha_dir"], exist_ok=True)
    ckpt = json.load(open(cfg["paths"]["coconut_ckpt_meta"], "r", encoding="utf-8"))["ckpt_path"]
    v_truth = torch.load(
        os.path.join(cfg["paths"]["vectors_dir"], "source_a_dom_shared.pt"),
        map_location=_dev(),
    )
    v_truth = v_truth / (v_truth.norm() + 1e-12)

    base_model, tokenizer = build_base_lm(
        cfg["model_id"],
        add_latent_tokens=True,
        load_model_path=ckpt,
    )

    theta = torch.nn.Parameter(torch.tensor(-3.89, device=_dev(), dtype=torch.float32))

    def alpha_fn():
        return cfg["phase3"]["alpha_max"] * torch.sigmoid(theta)

    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    model = SteeredCoconut(
        base_causallm=base_model,
        latent_token_id=latent_id,
        start_latent_id=start_id,
        end_latent_id=end_id,
        eos_token_id=tokenizer.eos_token_id,
        alpha_fn=alpha_fn,
        v_truth=v_truth,
        gamma=cfg["phase3"]["gamma"],
    ).to(_dev())

    for p in model.base_causallm.parameters():
        p.requires_grad = False

    val_base = get_dataset(cfg["paths"]["val_path"], tokenizer)
    cobj = SimpleNamespace(**{**cfg["curriculum"], **cfg})
    q_ds = get_question_latent_dataset(
        cfg["curriculum"]["max_latent_stage"],
        val_base,
        cobj,
        start_id,
        latent_id,
        end_id,
    )
    train_ds = get_cot_latent_dataset(
        scheduled_stage=cfg["curriculum"]["max_latent_stage"],
        base_dataset=val_base,
        configs=cobj,
        start_id=start_id,
        latent_id=latent_id,
        end_id=end_id,
        no_special_marker=False,
    )
    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collator)
    opt = torch.optim.Adam([theta], lr=cfg["phase3"]["lr"])

    best = {"alpha": float(alpha_fn().item()), "acc": -1.0, "step": -1}
    best_step = 0
    for step, batch in enumerate(loader):
        if step >= cfg["phase3"]["max_steps"]:
            break
        batch = {k: v.to(_dev()) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(**batch)
        l_ans = out.loss
        h_seq = getattr(model, "_h_seq", [])
        if h_seq:
            l_mag = -torch.stack([h.norm(dim=-1).mean() for h in h_seq]).mean()
        else:
            l_mag = torch.tensor(0.0, device=_dev())
        loss = l_ans + cfg["phase3"]["lambda_m"] * l_mag
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 10 == 0:
            acc = _eval_acc(model, tokenizer, q_ds, max_new_tokens=64)
            if acc > best["acc"]:
                best = {"alpha": float(alpha_fn().item()), "acc": float(acc), "step": step}
                best_step = step
            if step - best_step >= cfg["phase3"]["patience"]:
                break

    with open(os.path.join(cfg["paths"]["alpha_dir"], "best_alpha.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)


if __name__ == "__main__":
    run(_load_cfg(sys.argv[1]))
