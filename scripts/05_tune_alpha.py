import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.alpha_tune import run as run_alpha


def main():
    run_alpha(__import__("yaml").safe_load(open(sys.argv[1], "r", encoding="utf-8")))


if __name__ == "__main__":
    main()
