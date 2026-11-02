from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
import torch
from egrc.train import forward_episode, save_checkpoint, load_egrc_checkpoint
from egrc.model import EGRCConfig, EGRCModel


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 16)

    def forward(self, images):
        return self.projection(images)


class TrainingTests(unittest.TestCase):
    def test_rejects_unidentified_or_incomplete_main_checkpoints(self):
        model = EGRCModel(
            TinyBackbone(),
            EGRCConfig(dim=16, relation_dim=8, relation_hidden_dim=16, task_hidden_dim=8),
        )
        head, backbone = model.head.state_dict(), model.backbone.state_dict()
        cases = [
            {"head": head, "backbone": backbone},
            {"method": "EGRC", "format": "egrc-vit-s16-v1", "head": head},
            {"method": "EGRC", "format": "egrc-vit-s16-v1", "head": head, "backbone": None},
            {
                "head": head,
                "backbone": backbone,
                "args": {
                    "ablation_variant": "tc_eif",
                    "ablation_components": {"tc": True, "eif": True, "dmp": True, "cm": False},
                },
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.pth"
            for index, state in enumerate(cases):
                with self.subTest(case=index):
                    torch.save(state, path)
                    with self.assertRaises((ValueError, RuntimeError)):
                        load_egrc_checkpoint(path, model, "cpu")

    def test_valid_legacy_main_checkpoint_is_supported(self):
        model = EGRCModel(
            TinyBackbone(),
            EGRCConfig(dim=16, relation_dim=8, relation_hidden_dim=16, task_hidden_dim=8),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pth"
            torch.save(
                {
                    "head": model.head.state_dict(),
                    "backbone": model.backbone.state_dict(),
                    "args": {
                        "ablation_variant": "tc_eif",
                        "ablation_components": {"tc": True, "eif": True, "dmp": False, "cm": False},
                    },
                },
                path,
            )
            load_egrc_checkpoint(path, model, "cpu")

    def test_query_labels_do_not_enter_model_and_graph_logits_are_used(self):
        model = EGRCModel(
            TinyBackbone(),
            EGRCConfig(dim=16, relation_dim=8, relation_hidden_dim=16, task_hidden_dim=8),
        )
        args = Namespace(way=3, test_way=3, shot=1, query=2, backbone_chunk_size=32)
        images = torch.randn(9, 4)
        logits, aux, labels, all_labels = forward_episode(
            model, (images, torch.full((9,), -99)), args, torch.device("cpu"), True
        )
        torch.testing.assert_close(logits, aux["graph_logits"])
        self.assertEqual(labels.tolist(), [0, 1, 2, 0, 1, 2])
        self.assertEqual(all_labels[:3].tolist(), [0, 1, 2])

    def test_checkpoint_preserves_finetuned_backbone_and_head(self):
        model = EGRCModel(
            TinyBackbone(),
            EGRCConfig(dim=16, relation_dim=8, relation_hidden_dim=16, task_hidden_dim=8),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            save_checkpoint(
                path, model, None, None, 2, 0.8, Namespace(dataset="MiniImageNet", shot=1)
            )
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            with torch.no_grad():
                for value in model.parameters():
                    value.add_(1)
            load_egrc_checkpoint(path, model, "cpu")
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[key])


if __name__ == "__main__":
    unittest.main()
