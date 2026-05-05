import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.evaluate import run as run_eval


def main():
    run_eval(__import__("yaml").safe_load(open(sys.argv[1], "r", encoding="utf-8")))


if __name__ == "__main__":
    main()
