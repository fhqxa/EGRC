import csv
import os.path as osp

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .dataset_readers import resolve_dataset_root


DEFAULT_ROOT = "datasets"


class MiniImageNet(Dataset):
    def __init__(self, setname, args):
        configured_root = getattr(args, "dataset_root", None) or DEFAULT_ROOT
        self.root = resolve_dataset_root(
            configured_root,
            ["mini-imagenet", "miniimagenet", "mini"],
            validator=self._is_valid_root,
        )
        csv_path = self._find_split_csv(self.root, setname)
        self.image_path = osp.join(self.root, "images")
        class_dict = self._collect_samples_from_csv(csv_path, self.image_path)
        if not class_dict:
            raise RuntimeError(f"No valid classes found in {csv_path}")

        self.class_dict = class_dict
        self.wnids = sorted(class_dict.keys())
        self.data = []
        self.label = []
        for lb, wnid in enumerate(self.wnids):
            for path in class_dict[wnid]:
                self.data.append(path)
                self.label.append(lb)
        self.num_class = len(self.wnids)
        self.setname = setname

        image_size = 224
        self.transform_train = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
                ),
            ]
        )
        self.transform_val_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _is_valid_root(dataset_root):
        for mode in ("train", "val", "test"):
            for candidate in [
                osp.join(dataset_root, "split", f"{mode}.csv"),
                osp.join(dataset_root, f"{mode}.csv"),
            ]:
                if osp.exists(candidate):
                    return True
        return False

    @staticmethod
    def _find_split_csv(dataset_root, mode):
        for candidate in [
            osp.join(dataset_root, "split", f"{mode}.csv"),
            osp.join(dataset_root, f"{mode}.csv"),
        ]:
            if osp.exists(candidate):
                return candidate
        raise FileNotFoundError(f"Split CSV not found for {mode}: searched under {dataset_root}")

    @staticmethod
    def _collect_samples_from_csv(csv_file, image_root):
        class_dict = {}
        with open(csv_file, "r", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None:
                raise ValueError(f"CSV file is empty: {csv_file}")

            for row in reader:
                if len(row) < 2:
                    continue
                filename, class_name = row[0], row[1]
                img_path = osp.join(image_root, filename)
                if not osp.exists(img_path):
                    img_path = osp.join(image_root, class_name, filename)
                if osp.exists(img_path):
                    class_dict.setdefault(class_name, []).append(img_path)
        return class_dict

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        if not osp.exists(path):
            raise FileNotFoundError(
                f"MiniImageNet image not found: {path}. "
                f"Please check dataset_root; resolved image directory: {self.image_path}"
            )
        if self.setname == "train":
            image = self.transform_train(Image.open(path).convert("RGB"))
        else:
            image = self.transform_val_test(Image.open(path).convert("RGB"))
        return image, label
