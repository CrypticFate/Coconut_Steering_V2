import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import prepare_splits, read_json, write_json


def main():
    train_pool = read_json("data/coconut_format/gsm_train.json")
    test_pool = read_json("data/coconut_format/gsm_test.json")
    splits = prepare_splits(train_pool, test_pool, seed=42)
    os.makedirs("data/splits", exist_ok=True)
    for name, rows in splits.items():
        write_json(f"data/splits/{name}.json", rows)
        print(f"{name}: {len(rows)} examples")


if __name__ == "__main__":
    main()
