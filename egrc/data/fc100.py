import os
import os.path as osp

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .dataset_readers import list_image_files, resolve_dataset_root


DEFAULT_ROOT = "datasets"


class FC100(Dataset):
    def __init__(self, setname, args):
        configured_root = getattr(args, "dataset_root", None) or DEFAULT_ROOT
        self.root = resolve_dataset_root(
            configured_root,
            ["FC100", "fc100"],
            validator=self._is_valid_root,
        )
        class_dict = self._load_class_dict(self.root, setname)
        if not class_dict:
            raise RuntimeError(f"No valid classes found for FC100 {setname} split.")

        self.class_dict = class_dict
        self.classes = sorted(class_dict.keys())
        self.data = []
        self.label = []
        for idx, class_name in enumerate(self.classes):
            for image_path in class_dict[class_name]:
                self.data.append(image_path)
                self.label.append(idx)
        self.num_class = len(self.classes)
        self.setname = setname

        image_size = 224
        self.transform_train = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(image_size),
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
        for candidate in [
            osp.join(dataset_root, "train"),
            osp.join(dataset_root, "val"),
            osp.join(dataset_root, "test"),
            osp.join(dataset_root, "meta-train"),
            osp.join(dataset_root, "meta-val"),
            osp.join(dataset_root, "meta-test"),
        ]:
            if osp.isdir(candidate):
                return True
        return False

    @staticmethod
    def _load_class_dict(dataset_root, mode):
        class_dict = {}
        split_dir_candidates = [
            osp.join(dataset_root, mode),
            osp.join(dataset_root, f"meta-{mode}"),
        ]

        split_dir = None
        for candidate in split_dir_candidates:
            if osp.isdir(candidate):
                split_dir = candidate
                break
        if split_dir is None:
            raise FileNotFoundError(
                f"FC100 split directory not found for {mode}: searched under {dataset_root}"
            )

        for class_name in sorted(os.listdir(split_dir)):
            class_dir = osp.join(split_dir, class_name)
            if not osp.isdir(class_dir):
                continue
            for image_file in list_image_files(class_dir):
                class_dict.setdefault(class_name, []).append(osp.join(class_dir, image_file))
        return class_dict

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        if not osp.exists(path):
            raise FileNotFoundError(f"FC100 image not found: {path}")
        if self.setname == "train":
            image = self.transform_train(Image.open(path).convert("RGB"))
        else:
            image = self.transform_val_test(Image.open(path).convert("RGB"))
        return image, label
