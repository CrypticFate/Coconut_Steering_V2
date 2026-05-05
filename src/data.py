import json
import os
import random
from typing import Dict, List


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_answer(x: str) -> str:
    return " ".join(str(x).strip().lower().split())


def extract_answer_after_marker(text: str, marker: str = "### "):
    if marker in text:
        return text.split(marker)[-1].strip().split("\n")[0]
    return text.strip().split("\n")[-1]


def prepare_splits(train_pool: List[Dict], test_pool: List[Dict], seed: int = 42):
    random.seed(seed)
    random.shuffle(train_pool)
    random.shuffle(test_pool)
    train_pool = train_pool[:500]
    test_pool = test_pool[:100]

    for i, ex in enumerate(train_pool):
        ex["qid"] = f"train_{i:05d}"
    for i, ex in enumerate(test_pool):
        ex["qid"] = f"test_{i:05d}"

    n_train_full = int(0.60 * len(train_pool))
    n_steer = int(0.10 * len(train_pool))
    train_full = train_pool[:n_train_full]
    steer = train_pool[n_train_full : n_train_full + n_steer]
    val = train_pool[n_train_full + n_steer :]

    random.shuffle(train_full)
    n_int_val = int(0.10 * len(train_full))
    train_internal_val = train_full[:n_int_val]
    train = train_full[n_int_val:]

    splits = {
        "train": train,
        "train_internal_val": train_internal_val,
        "steer": steer,
        "val": val,
        "test": test_pool,
    }
    _assert_disjoint(splits)
    return splits


def _assert_disjoint(splits: Dict[str, List[Dict]]):
    seen = {}
    for name, rows in splits.items():
        qids = {r["qid"] for r in rows}
        for other_name, other_qids in seen.items():
            if not qids.isdisjoint(other_qids):
                raise ValueError(f"Split overlap detected between {name} and {other_name}")
        seen[name] = qids
