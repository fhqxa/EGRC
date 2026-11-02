"""Plan or run the eight main EGRC experiments."""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from egrc.config import parse_args


ROOT = Path(__file__).resolve().parent
DATASETS = ("mini", "tiered", "cifarfs", "fc100")


def build_jobs(datasets=DATASETS, shots=(1, 5), seeds=(42,), run_tag="", trainer_args=()):
    if run_tag and not re.fullmatch(r"[A-Za-z0-9_-]+", run_tag):
        raise ValueError("run-tag must contain only ASCII letters, digits, hyphens, or underscores")
    jobs = []
    for dataset in dict.fromkeys(datasets):
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        for shot in dict.fromkeys(shots):
            for seed in dict.fromkeys(seeds):
                name = f"egrc-{dataset}-5way-{shot}shot-seed{seed}"
                if run_tag:
                    name += f"-{run_tag}"
                arguments = [
                    "--config",
                    str(ROOT / "configs" / f"{dataset}_{shot}shot.json"),
                    "--seed",
                    str(seed),
                    "--exp",
                    name,
                    *trainer_args,
                ]
                args = parse_args(arguments)
                jobs.append(
                    {
                        "args": args,
                        "command": [sys.executable, "-u", str(ROOT / "train.py"), *arguments],
                    }
                )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--shots", nargs="+", type=int, choices=(1, 5), default=[1, 5])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--run-tag", default="")
    parser.add_argument(
        "--execute", action="store_true", help="Train; default only prints the plan"
    )
    args, trainer_args = parser.parse_known_args()
    if any(
        arg.split("=")[0] in ("--config", "--exp", "--seed", "--shot", "--dataset")
        for arg in trainer_args
    ):
        parser.error(
            "Use queue options to select datasets, shots, and seeds; use --run-tag for new experiment names"
        )
    jobs = build_jobs(args.datasets, args.shots, args.seeds, args.run_tag, trainer_args)
    print(f"{len(jobs)} main EGRC runs; training={args.execute}", flush=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    for job in jobs:
        selected = job["args"]
        print(
            f"{selected.exp}: lambda={selected.edge_loss_weight}, "
            f"chunk={selected.backbone_chunk_size}",
            flush=True,
        )
        print(shlex.join(job["command"]), flush=True)
        if args.execute:
            subprocess.run(job["command"], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
