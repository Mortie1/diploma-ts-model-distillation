from __future__ import annotations

import argparse
import subprocess


def run_cmd(cmd: list[str]):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["classification", "forecasting"], required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    run_cmd(["python3", "train.py", f"-cn={args.config}", f"task={args.task}"])
    run_cmd(["python3", "inference.py", f"-cn={args.task}_inference"])


if __name__ == "__main__":
    main()
