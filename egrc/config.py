"""Main-experiment configuration and paths (standard library only)."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "MiniImageNet": ("mini", "miniimagenet", "miniimagenet_checkpoint1600.pth"),
    "tieredImageNet": ("tiered", "tieredimagenet", "tieredimagenet_checkpoint0800.pth"),
    "CIFARFS": ("cifarfs", "cifar-fs", "cifar-fs_checkpoint1600.pth"),
    "FC100": ("fc100", "fc100", "fc100_checkpoint1600.pth"),
}


def build_parser():
    parser = argparse.ArgumentParser(description="Main EGRC training and evaluation.")
    parser.add_argument("--config")
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="MiniImageNet")
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--init-weights")
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--test-way", type=int, default=5)
    parser.add_argument("--shot", type=int, choices=(1, 5), default=1)
    parser.add_argument("--query", type=int, default=15)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--val-episodes", type=int, default=500)
    parser.add_argument("--test-episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-episode-seed", type=int)
    parser.add_argument("--val-episode-seed", type=int, default=100042)
    parser.add_argument("--test-episode-seed", type=int, default=200042)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--backbone-chunk-size", type=int, default=32)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--backbone-weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--edge-loss-weight", type=float)
    parser.add_argument("--edge-kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--positive-topk", type=int, default=10)
    parser.add_argument("--evidence-max", type=float, default=20.0)
    parser.add_argument("--update-scale-max", type=float, default=0.2)
    parser.add_argument("--proto-temperature", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--classifier-dropout", type=float, default=0.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--vit-gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-root", default="logs")
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--exp")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    known, _ = probe.parse_known_args(argv)
    parser = build_parser()
    if known.config:
        path = Path(known.config)
        path = path if path.is_absolute() else ROOT / path
        defaults = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(defaults, dict):
            parser.error("The configuration must be one JSON object")
        invalid = set(defaults) - {action.dest for action in parser._actions}
        if invalid:
            parser.error("Unknown configuration keys: " + ", ".join(sorted(invalid)))
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.dataset not in DATASETS or args.shot not in (1, 5):
        parser.error("Unsupported dataset or shot in configuration")
    if args.edge_loss_weight is None:
        args.edge_loss_weight = 0.01 if args.shot == 1 else 0.05
    if args.train_episode_seed is None:
        args.train_episode_seed = args.seed
    if (
        min(
            args.way,
            args.test_way,
            args.query,
            args.max_epoch,
            args.train_episodes,
            args.val_episodes,
            args.test_episodes,
            args.backbone_chunk_size,
        )
        < 1
    ):
        parser.error("Episode counts, class counts, query count, and chunk size must be positive")
    if args.num_workers < 0 or args.log_every < 0 or args.edge_loss_weight < 0:
        parser.error("Worker count, log interval, and loss weight must be nonnegative")
    return args


def _project_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def resolve_paths(args):
    args.dataset_root = str(_project_path(args.dataset_root))
    short, folder, name = DATASETS[args.dataset]
    args.init_weights = str(
        _project_path(args.init_weights or Path("initialization") / folder / name)
    )
    if not Path(args.init_weights).is_file():
        raise FileNotFoundError(
            f"Pretrained ViT checkpoint not found: {args.init_weights}. "
            "See initialization/README.md or use --init-weights."
        )
    args.exp = args.exp or f"egrc-{short}-{args.way}way-{args.shot}shot-seed{args.seed}"
    if (
        Path(args.exp).name != args.exp
        or "/" in args.exp
        or "\\" in args.exp
        or args.exp in (".", "..")
    ):
        raise ValueError("Experiment name must be one directory name")
    args.log_path = str(_project_path(args.log_root) / args.exp)
    args.checkpoint_path = str(_project_path(args.checkpoint_root) / args.exp)
    if args.checkpoint:
        args.checkpoint = str(_project_path(args.checkpoint))
    if args.eval_only and (not args.checkpoint or not Path(args.checkpoint).is_file()):
        raise FileNotFoundError("--eval-only requires an existing --checkpoint file")
    if not args.eval_only:
        for path, allowed in (
            (Path(args.log_path), {"resolved_args.json"}),
            (Path(args.checkpoint_path), set()),
        ):
            if path.exists() and any(
                not item.is_file() or item.name not in allowed for item in path.iterdir()
            ):
                raise FileExistsError(
                    f"Refusing to overwrite existing training records: {path}. "
                    "Choose a new --exp, or select the remaining queue items."
                )
    Path(args.log_path).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)
    return args
