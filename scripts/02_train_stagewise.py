import json
import os
import sys
import argparse

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.train_phase1 import run_coconut, run_cot


def _build_cot_cfg(master_cfg):
    return {
        "model_id": master_cfg["model_id"],
        "seed": master_cfg["seed"],
        "train_path": master_cfg["paths"]["train_path"],
        "val_path": master_cfg["paths"]["train_internal_val_path"],
        "cot": True,
        "coconut": False,
        "no_thoughts": False,
        "no_cot": False,
        "num_epochs": master_cfg["phase1a"]["num_epochs"],
        "batch_size_training": master_cfg["phase1a"]["batch_size_training"],
        "gradient_accumulation_steps": master_cfg["phase1a"]["gradient_accumulation_steps"],
        "lr": master_cfg["phase1a"]["lr"],
        "weight_decay": master_cfg["phase1a"]["weight_decay"],
        "save_only_improve": master_cfg["phase1a"]["save_only_improve"],
        "save_path": "outputs/checkpoints",
        "name": "cot_dryrun",
        "c_thought": master_cfg["curriculum"]["c_thought"],
        "epochs_per_stage": master_cfg["curriculum"]["epochs_per_stage"],
        "max_latent_stage": master_cfg["curriculum"]["max_latent_stage"],
        "pad_latent_to_max": master_cfg["curriculum"]["pad_latent_to_max"],
        "uniform_prob": master_cfg["curriculum"]["uniform_prob"],
        "gradient_checkpointing": master_cfg.get("memory", {}).get("gradient_checkpointing", False),
    }


def _build_coconut_cfg(master_cfg, cot_ckpt_path):
    return {
        "model_id": master_cfg["model_id"],
        "seed": master_cfg["seed"],
        "train_path": master_cfg["paths"]["train_path"],
        "val_path": master_cfg["paths"]["train_internal_val_path"],
        "load_model_path": cot_ckpt_path,
        "cot": False,
        "coconut": True,
        "no_thoughts": False,
        "no_cot": False,
        "num_epochs": master_cfg["phase1b"]["num_epochs"],
        "resume": master_cfg["phase1b"]["resume"],
        "batch_size_training": master_cfg["phase1b"]["batch_size_training"],
        "gradient_accumulation_steps": master_cfg["phase1b"]["gradient_accumulation_steps"],
        "lr": master_cfg["phase1b"]["lr"],
        "weight_decay": master_cfg["phase1b"]["weight_decay"],
        "reset_optimizer": master_cfg["phase1b"]["reset_optimizer"],
        "save_only_improve": master_cfg["phase1b"]["save_only_improve"],
        "stage_plan": master_cfg["phase1b"].get("stage_plan"),
        "save_path": "outputs/checkpoints",
        "name": "coconut_dryrun",
        "c_thought": master_cfg["curriculum"]["c_thought"],
        "epochs_per_stage": master_cfg["curriculum"]["epochs_per_stage"],
        "max_latent_stage": master_cfg["curriculum"]["max_latent_stage"],
        "pad_latent_to_max": master_cfg["curriculum"]["pad_latent_to_max"],
        "uniform_prob": master_cfg["curriculum"]["uniform_prob"],
        "gradient_checkpointing": master_cfg.get("memory", {}).get("gradient_checkpointing", False),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Override model id from config for quick swaps on local laptops.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config_path, "r", encoding="utf-8"))
    if args.model_id:
        cfg["model_id"] = args.model_id
        print(f"Overriding model_id with: {cfg['model_id']}")

    logs_dir = cfg["paths"].get("logs_dir", "outputs/logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "phase1_stagewise.jsonl")

    print("=== Stage 1: Full CoT training ===")
    cot_cfg = _build_cot_cfg(cfg)
    run_cot(cot_cfg, log_path=log_path, stage_label="stage1_full_cot")

    cot_best = json.load(open(cfg["paths"]["cot_ckpt_meta"], "r", encoding="utf-8"))
    cot_ckpt_path = cot_best["ckpt_path"]
    print(f"Best CoT checkpoint: {cot_ckpt_path}")

    print("=== Stage 2+: Coconut architecture stagewise training ===")
    coconut_cfg = _build_coconut_cfg(cfg, cot_ckpt_path)
    run_coconut(coconut_cfg, log_path=log_path)

    print("Stagewise phase-1 training complete.")
    print(f"Logs saved at: {log_path}")


if __name__ == "__main__":
    main()
