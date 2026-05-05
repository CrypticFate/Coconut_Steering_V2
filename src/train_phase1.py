import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "external/coconut")
from dataset import MyCollator, get_cot_latent_dataset, get_dataset, get_question_latent_dataset

from src.coconut_utils import build_base_lm, wrap_in_coconut


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_final_answer(text: str):
    if "### " in text:
        tail = text.split("### ")[-1]
    else:
        tail = text
    tail = tail.strip().split("\n")[0]
    m = re.search(r"[-+]?\d[\d,]*\.?\d*", tail)
    return m.group(0).replace(",", "") if m else tail.strip().lower()


def _append_log(log_path, payload):
    if not log_path:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    row = {"ts_utc": datetime.now(timezone.utc).isoformat(), **payload}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _build_coconut_stage_plan(cfg):
    if cfg.get("stage_plan"):
        return cfg["stage_plan"]
    resume = cfg["resume"]
    end_epoch = cfg["resume"] + cfg["num_epochs"] - 1
    eps_per_stage = cfg["epochs_per_stage"]
    max_stage = cfg["max_latent_stage"]
    groups = {}
    for epoch in range(resume, end_epoch + 1):
        stage = min(max_stage, epoch // eps_per_stage)
        groups.setdefault(stage, []).append(epoch)
    plan = []
    for stage, epochs in sorted(groups.items(), key=lambda kv: kv[0]):
        plan.append({"stage": int(stage), "epochs": len(epochs)})
    return plan


def evaluate_exact_match(model, tokenizer, val_rows, max_new_tokens=200):
    model.eval()
    correct = 0
    total = max(1, len(val_rows))
    for ex in val_rows:
        prompt = ex["question"] + "\nLet's think step by step.\n### "
        toks = tokenizer(prompt, return_tensors="pt").to(_dev())
        with torch.no_grad():
            out = model.generate(**toks, max_new_tokens=max_new_tokens, do_sample=False)
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = _extract_final_answer(pred)
        gold = _extract_final_answer(ex["answer"])
        correct += int(pred == gold)
    return correct / total


def evaluate_coconut_exact_match(model, tokenizer, val_dataset, cfg):
    model.eval()
    correct = 0
    total = max(1, len(val_dataset))
    for item in val_dataset:
        input_ids = item["input_ids"].unsqueeze(0).to(_dev())
        attention_mask = item["attention_mask"].unsqueeze(0).to(_dev())
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                synced_gpus=False,
            )
        pred = _extract_final_answer(tokenizer.decode(out_ids[0], skip_special_tokens=True))
        gold = _extract_final_answer(item["answer"])
        correct += int(pred == gold)
    return correct / total


def run_cot(cfg, log_path=None, stage_label="cot"):
    _seed(cfg["seed"])
    model, tokenizer = build_base_lm(cfg["model_id"], add_latent_tokens=False)
    model.to(_dev())
    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    train_ds = get_dataset(cfg["train_path"], tokenizer)
    val_ds = get_dataset(cfg["val_path"], tokenizer)
    with open(cfg["val_path"], "r", encoding="utf-8") as f:
        val_rows_raw = json.load(f)
    collator = MyCollator(tokenizer, latent_id=None, label_pad_token_id=-100)
    cobj = SimpleNamespace(**cfg)

    def build_epoch_ds(epoch):
        return get_cot_latent_dataset(
            scheduled_stage=0,
            base_dataset=train_ds,
            configs=cobj,
            start_id=None,
            latent_id=None,
            end_id=None,
            no_special_marker=True,
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    best_val_acc = -1.0
    out_dir = os.path.join(cfg["save_path"], cfg["name"])
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(cfg["num_epochs"]):
        epoch_ds = build_epoch_ds(epoch)
        loader = DataLoader(epoch_ds, batch_size=cfg["batch_size_training"], shuffle=True, collate_fn=collator)
        model.train()
        running_loss = 0.0
        n_steps = 0
        pbar = tqdm(loader, desc=f"[{stage_label}] epoch {epoch}", leave=True)
        for step, batch in enumerate(pbar):
            batch = {k: v.to(_dev()) if torch.is_tensor(v) else v for k, v in batch.items()}
            step_loss = model(**batch).loss
            loss = step_loss / cfg["gradient_accumulation_steps"]
            loss.backward()
            running_loss += step_loss.detach().item()
            n_steps += 1
            pbar.set_postfix({"loss": f"{(running_loss/max(1,n_steps)):.4f}"})
            if (step + 1) % cfg["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()

        val_acc = evaluate_exact_match(model, tokenizer, val_rows_raw, max_new_tokens=200)
        mean_train_loss = running_loss / max(1, n_steps)
        print(f"[{stage_label}] epoch={epoch} train_loss={mean_train_loss:.4f} val_acc={val_acc:.4f}")
        _append_log(
            log_path,
            {
                "phase": "phase1a_cot",
                "stage_label": stage_label,
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "val_acc": val_acc,
            },
        )
        if (not cfg["save_only_improve"]) or (val_acc > best_val_acc):
            ckpt = os.path.join(out_dir, f"checkpoint_{epoch}")
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                with open(os.path.join(out_dir, "best_val_acc.json"), "w", encoding="utf-8") as f:
                    json.dump({"best_epoch": epoch, "val_acc": val_acc, "ckpt_path": ckpt}, f, indent=2)


def run_coconut(cfg, log_path=None):
    _seed(cfg["seed"])
    load_path = cfg.get("load_model_path")
    if load_path and load_path != "None":
        base_model, tokenizer = build_base_lm(
            cfg["model_id"],
            add_latent_tokens=True,
            load_model_path=load_path,
        )
    else:
        base_model, tokenizer = build_base_lm(cfg["model_id"], add_latent_tokens=True)
    model = wrap_in_coconut(base_model, tokenizer).to(_dev())
    if cfg.get("gradient_checkpointing", False) and hasattr(model.base_causallm, "gradient_checkpointing_enable"):
        model.base_causallm.gradient_checkpointing_enable()
        if hasattr(model.base_causallm.config, "use_cache"):
            model.base_causallm.config.use_cache = False

    train_ds = get_dataset(cfg["train_path"], tokenizer)
    val_ds = get_dataset(cfg["val_path"], tokenizer)
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    cobj = SimpleNamespace(**cfg)

    def build_epoch_ds(epoch):
        return get_cot_latent_dataset(
            scheduled_stage=epoch // cfg["epochs_per_stage"],
            base_dataset=train_ds,
            configs=cobj,
            start_id=start_id,
            latent_id=latent_id,
            end_id=end_id,
            no_special_marker=False,
        )

    val_q_ds = get_question_latent_dataset(
        cfg["max_latent_stage"],
        val_ds,
        cobj,
        start_id,
        latent_id,
        end_id,
    )
    out_dir = os.path.join(cfg["save_path"], cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    best_val_acc = -1.0

    stage_plan = _build_coconut_stage_plan(cfg)
    epoch_cursor = cfg["resume"]

    for plan_idx, plan in enumerate(stage_plan):
        stage = int(plan["stage"])
        n_epochs = int(plan["epochs"])
        print(f"\n===== Coconut Stage {stage} | epochs={n_epochs} | start_epoch={epoch_cursor} =====")
        _append_log(
            log_path,
            {
                "phase": "phase1b_coconut",
                "event": "stage_start",
                "stage": stage,
                "epochs": n_epochs,
                "start_epoch": epoch_cursor,
                "stage_plan_index": plan_idx,
            },
        )

        for _ in range(n_epochs):
            epoch = epoch_cursor
            epoch_cursor += 1
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=cfg["weight_decay"]
            )
            epoch_ds = build_epoch_ds(epoch)
            loader = DataLoader(epoch_ds, batch_size=cfg["batch_size_training"], shuffle=True, collate_fn=collator)
            model.train()
            running_loss = 0.0
            n_steps = 0
            pbar = tqdm(loader, desc=f"[coconut stage {stage}] epoch {epoch}", leave=True)
            for step, batch in enumerate(pbar):
                batch = {k: v.to(_dev()) if torch.is_tensor(v) else v for k, v in batch.items()}
                step_loss = model(**batch).loss
                loss = step_loss / cfg["gradient_accumulation_steps"]
                loss.backward()
                running_loss += step_loss.detach().item()
                n_steps += 1
                pbar.set_postfix({"loss": f"{(running_loss/max(1,n_steps)):.4f}"})
                if (step + 1) % cfg["gradient_accumulation_steps"] == 0:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            val_acc = evaluate_coconut_exact_match(model, tokenizer, val_q_ds, cfg)
            mean_train_loss = running_loss / max(1, n_steps)
            print(
                f"[coconut stage {stage}] epoch={epoch} train_loss={mean_train_loss:.4f} val_acc={val_acc:.4f}"
            )
            _append_log(
                log_path,
                {
                    "phase": "phase1b_coconut",
                    "stage": stage,
                    "epoch": epoch,
                    "train_loss": mean_train_loss,
                    "val_acc": val_acc,
                },
            )
            if (not cfg["save_only_improve"]) or (val_acc > best_val_acc):
                ckpt = os.path.join(out_dir, f"checkpoint_{epoch}")
                model.base_causallm.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    with open(os.path.join(out_dir, "best_val_acc.json"), "w", encoding="utf-8") as f:
                        json.dump({"best_epoch": epoch, "val_acc": val_acc, "ckpt_path": ckpt}, f, indent=2)


if __name__ == "__main__":
    cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
    if cfg.get("cot", False):
        run_cot(cfg)
    elif cfg.get("coconut", False):
        run_coconut(cfg)
    else:
        raise ValueError("Config must enable either cot=true or coconut=true.")
