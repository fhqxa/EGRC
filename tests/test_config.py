import json
from pathlib import Path
import tempfile
import unittest
from egrc.config import parse_args, resolve_paths


class ConfigTests(unittest.TestCase):
    def test_default_losses_follow_shot_and_cli_overrides_config(self):
        for shot, weight in ((1, 0.01), (5, 0.05)):
            args = parse_args(["--shot", str(shot)])
            self.assertEqual(args.edge_loss_weight, weight)
            self.assertEqual(args.backbone_chunk_size, 32)
            self.assertEqual((args.val_episode_seed, args.test_episode_seed), (100042, 200042))
        args = parse_args(["--config", "configs/mini_1shot.json", "--seed", "43"])
        self.assertEqual(args.seed, 43)
        self.assertEqual(args.train_episode_seed, 43)
        self.assertEqual((args.val_episode_seed, args.test_episode_seed), (100042, 200042))

    def test_only_eight_main_configs_with_correct_hyperparameters(self):
        folder = Path(__file__).resolve().parents[1] / "configs"
        paths = sorted(folder.glob("*.json"))
        self.assertEqual(len(paths), 8)
        for path in paths:
            config = json.loads(path.read_text(encoding="utf-8"))
            args = parse_args(["--config", str(path)])
            self.assertEqual(args.edge_loss_weight, 0.01 if args.shot == 1 else 0.05)
            self.assertEqual(args.backbone_chunk_size, 32)
            self.assertEqual(
                (args.max_epoch, args.train_episodes, args.val_episodes, args.test_episodes),
                (100, 100, 500, 1000),
            )
            self.assertFalse(any(("ablation" in str(v).lower() for v in config.values())))

    def test_existing_training_records_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "init.pth"
            weights.write_bytes(b"path-check fixture")
            arguments = [
                "--init-weights",
                str(weights),
                "--log-root",
                str(root / "logs"),
                "--checkpoint-root",
                str(root / "checkpoints"),
                "--exp",
                "test",
            ]
            args = resolve_paths(parse_args(arguments))
            (Path(args.log_path) / "metrics.jsonl").write_text("{}\n")
            with self.assertRaises(FileExistsError):
                resolve_paths(parse_args(arguments))
            with self.assertRaises(ValueError):
                resolve_paths(parse_args(arguments[:-1] + ["../escape"]))


if __name__ == "__main__":
    unittest.main()
