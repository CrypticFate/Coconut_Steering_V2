import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.train_phase1 import run_coconut


def main():
    cfg_path = sys.argv[1]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_coconut(cfg, log_path="outputs/logs/phase1b_coconut.jsonl")


if __name__ == "__main__":
    main()
