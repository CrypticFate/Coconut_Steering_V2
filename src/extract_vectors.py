import json
import os
import random
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, "external/coconut")
from dataset import get_dataset, get_question_latent_dataset

from src.data import extract_answer_after_marker, normalize_answer
from src.coconut_utils import build_base_lm, wrap_in_coconut


def _load_cfg(path):
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(cfg):
    os.makedirs(cfg["paths"]["vectors_dir"], exist_ok=True)
    ckpt_path = json.load(open(cfg["paths"]["coconut_ckpt_meta"], "r", encoding="utf-8"))["ckpt_path"]
    base_model, tokenizer = build_base_lm(
        cfg["model_id"],
        add_latent_tokens=True,
        load_model_path=ckpt_path,
    )
    model = wrap_in_coconut(base_model, tokenizer)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    captured = {"h": []}

    def hook(_, __, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"].append(h.detach().cpu())

    hook_target = model.base_causallm.base_model.model.model.norm
    handle = hook_target.register_forward_hook(hook)

    steer_ds = get_dataset(cfg["paths"]["steer_path"], tokenizer)
    cobj = SimpleNamespace(**{**cfg["curriculum"], **cfg})
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    q_ds = get_question_latent_dataset(
        cfg["curriculum"]["max_latent_stage"],
        steer_ds,
        cobj,
        start_id,
        latent_id,
        end_id,
    )

    h_pos = defaultdict(list)
    h_neg = defaultdict(list)
    n_expected = cfg["curriculum"]["max_latent_stage"] * cfg["curriculum"]["c_thought"]

    with torch.no_grad():
        for item in q_ds:
            for _ in range(cfg["phase2"]["N_samples_per_question"]):
                captured["h"].clear()
                out = model.generate(
                    input_ids=item["input_ids"].unsqueeze(0).cuda(),
                    attention_mask=item["attention_mask"].unsqueeze(0).cuda(),
                    max_new_tokens=cfg["phase2"]["max_new_tokens"],
                    do_sample=True,
                    temperature=cfg["phase2"]["temperature"],
                    synced_gpus=False,
                )
                decoded = tokenizer.decode(out[0], skip_special_tokens=False)
                pred = extract_answer_after_marker(decoded, marker="### ")
                gold = item["answer"]
                is_correct = normalize_answer(pred) == normalize_answer(gold)
                for t in range(min(n_expected, len(captured["h"]) - 1)):
                    h_t = captured["h"][t][:, -1, :].squeeze(0)
                    (h_pos if is_correct else h_neg)[t].append((h_t, item["qid"]))

    vectors = {}
    for t in range(n_expected):
        pos = [x[0].numpy() for x in h_pos[t]]
        neg = [x[0].numpy() for x in h_neg[t]]
        if len(pos) == 0 or len(neg) == 0:
            continue
        v = np.mean(pos, axis=0) - np.mean(neg, axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)
        vectors[f"dom_t{t}"] = torch.tensor(v, dtype=torch.float32)

    if vectors:
        shared = torch.stack(list(vectors.values()), dim=0).mean(dim=0)
        shared = shared / (shared.norm() + 1e-12)
        torch.save(shared, os.path.join(cfg["paths"]["vectors_dir"], "source_a_dom_shared.pt"))

    # Linear probe (by-question split)
    all_rows = []
    for t in range(n_expected):
        for h, qid in h_pos[t]:
            all_rows.append((qid, h.numpy(), 1))
        for h, qid in h_neg[t]:
            all_rows.append((qid, h.numpy(), 0))
    random.shuffle(all_rows)
    qids = sorted({r[0] for r in all_rows})
    split = int(0.8 * len(qids))
    train_qids = set(qids[:split])
    x_train = np.array([r[1] for r in all_rows if r[0] in train_qids])
    y_train = np.array([r[2] for r in all_rows if r[0] in train_qids])
    x_test = np.array([r[1] for r in all_rows if r[0] not in train_qids])
    y_test = np.array([r[2] for r in all_rows if r[0] not in train_qids])
    if len(x_train) and len(x_test):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(x_train, y_train)
        acc = accuracy_score(y_test, clf.predict(x_test))
    else:
        acc = 0.0

    summary = {
        "n_expected_latents": n_expected,
        "h_pos_counts": {str(k): len(v) for k, v in h_pos.items()},
        "h_neg_counts": {str(k): len(v) for k, v in h_neg.items()},
        "probe_acc": float(acc),
    }
    with open(os.path.join(cfg["paths"]["vectors_dir"], "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    handle.remove()


if __name__ == "__main__":
    cfg = _load_cfg(sys.argv[1])
    run(cfg)
