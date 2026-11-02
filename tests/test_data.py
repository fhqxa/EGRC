import unittest
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
from PIL import Image
import torch
from egrc.data.samplers import CategoriesSampler
from egrc.data.loaders import build_dataset


class DataTests(unittest.TestCase):
    def test_all_readers_load_local_files_with_original_image_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, folder in (
                ("MiniImageNet", "miniimagenet"),
                ("tieredImageNet", "tieredimagenet"),
                ("CIFARFS", "cifar-fs"),
                ("FC100", "fc100"),
            ):
                for split in ("train", "val", "test"):
                    rows = []
                    for label in range(2):
                        for sample in range(3):
                            filename = f"{split}_{label}_{sample}.png"
                            if name == "MiniImageNet":
                                path = root / folder / "images" / filename
                                rows.append((filename, f"{split}_{label}"))
                            else:
                                path = root / folder / split / f"{split}_{label}" / filename
                            path.parent.mkdir(parents=True, exist_ok=True)
                            Image.new("RGB", (32, 32), (label * 80, sample * 40, 100)).save(path)
                    if name == "MiniImageNet":
                        with (root / folder / f"{split}.csv").open("w", newline="") as handle:
                            writer = csv.writer(handle)
                            writer.writerow(("filename", "label"))
                            writer.writerows(rows)
                args = SimpleNamespace(dataset=name, dataset_root=str(root))
                for split in ("train", "val", "test"):
                    dataset = build_dataset(args)(split, args)
                    image, label = dataset[0]
                    self.assertEqual((len(dataset), dataset.num_class), (6, 2))
                    self.assertEqual(tuple(image.shape), (3, 224, 224))
                    self.assertEqual(label, 0)
                    self.assertTrue(torch.isfinite(image).all())

    def test_repeated_validation_and_changing_training_episodes_use_shot_major_order(self):
        labels = [i for i in range(5) for _ in range(12)]
        val = CategoriesSampler(labels, 3, 3, 7, seed=100042, repeat_manifest=True)
        first, repeated = (list(val), list(val))
        for batch, again in zip(first, repeated):
            self.assertTrue(torch.equal(batch, again))
            selected = torch.tensor(labels)[batch].reshape(7, 3)
            self.assertTrue(torch.equal(selected, selected[0].expand_as(selected)))
            self.assertEqual(len(set(batch.tolist())), 21)
        train = CategoriesSampler(labels, 3, 3, 7, seed=42, repeat_manifest=False)
        self.assertFalse(torch.equal(list(train)[0], list(train)[0]))


if __name__ == "__main__":
    unittest.main()
