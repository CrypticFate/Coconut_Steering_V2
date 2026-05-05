import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.train_phase1 import run_cot


def main():
    cfg_path = sys.argv[1]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_cot(cfg, log_path="outputs/logs/phase1a_cot.jsonl", stage_label="cot")


if __name__ == "__main__":
    main()
