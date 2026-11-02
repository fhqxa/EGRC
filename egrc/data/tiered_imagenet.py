import os
import os.path as osp
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .dataset_readers import list_image_files, resolve_dataset_root


DEFAULT_ROOT = "datasets"


class TieredImagenet(Dataset):
    def __init__(self, setname, args):
        configured_root = getattr(args, "dataset_root", None) or DEFAULT_ROOT
        self.root = resolve_dataset_root(
            configured_root,
            ["tiered_imagenet", "tiered-imagenet", "tiered", "tieredimagenet", "TieredImageNet"],
            validator=self._is_valid_root,
        )
        class_dict = self._load_class_dict(self.root, setname)
        if not class_dict:
            raise RuntimeError(f"No valid classes found for tiered-imagenet {setname} split.")

        self.class_dict = class_dict
        self.wnids = sorted(class_dict.keys())
        self.data = []
        self.label = []
        for lb, wnid in enumerate(self.wnids):
            for sample in class_dict[wnid]:
                self.data.append(sample)
                self.label.append(lb)
        self.num_class = len(self.wnids)
        self.setname = setname

        image_size = 224
        self.transform_train = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                    np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]),
                ),
            ]
        )
        self.transform_val_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                    np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]),
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _is_valid_root(dataset_root):
        for mode in ("train", "val", "test"):
            if TieredImagenet._find_split_dir(dataset_root, mode) is not None:
                return True
            if TieredImagenet._find_npz_and_pkl(dataset_root, mode) is not None:
                return True
        return False

    @staticmethod
    def _load_class_dict(dataset_root, mode):
        class_dict = {}
        split_dir = TieredImagenet._find_split_dir(dataset_root, mode)
        npz_pkl_pair = TieredImagenet._find_npz_and_pkl(dataset_root, mode)

        if split_dir is not None:
            for root, _, _ in os.walk(split_dir):
                try:
                    image_files = list_image_files(root)
                except (FileNotFoundError, NotADirectoryError, PermissionError):
                    continue
                if not image_files:
                    continue
                class_name = osp.basename(root)
                for image_file in image_files:
                    class_dict.setdefault(class_name, []).append(osp.join(root, image_file))
            return class_dict

        if npz_pkl_pair is not None:
            npz_path, pkl_path = npz_pkl_pair
            with open(npz_path, "rb") as handle:
                image_data = np.load(handle)["images"]
            with open(pkl_path, "rb") as handle:
                unpickler = pickle._Unpickler(handle)
                unpickler.encoding = "latin1"
                label_data = unpickler.load()
            labels = label_data["labels"]
            for sample, label in zip(image_data, labels):
                class_dict.setdefault(int(label), []).append(sample)
            return class_dict

        raise FileNotFoundError(
            f"tiered-imagenet data not found under {dataset_root}. "
            f"Expected a train/val/test split directory or *_images.npz + *_labels.pkl under that dataset root."
        )

    @staticmethod
    def _find_split_dir(dataset_root, mode):
        direct_candidates = [
            osp.join(dataset_root, mode),
            osp.join(dataset_root, "images", mode),
        ]
        for candidate in direct_candidates:
            if osp.isdir(candidate):
                return candidate

        try:
            children = sorted(os.listdir(dataset_root))
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return None

        for child in children:
            child_dir = osp.join(dataset_root, child)
            if not osp.isdir(child_dir):
                continue
            nested_candidate = osp.join(child_dir, mode)
            if osp.isdir(nested_candidate):
                return nested_candidate
        return None

    @staticmethod
    def _find_npz_and_pkl(dataset_root, mode):
        candidate_roots = [dataset_root]
        try:
            candidate_roots.extend(
                osp.join(dataset_root, child)
                for child in sorted(os.listdir(dataset_root))
                if osp.isdir(osp.join(dataset_root, child))
            )
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            pass

        for root in candidate_roots:
            npz_path = osp.join(root, f"{mode}_images.npz")
            pkl_path = osp.join(root, f"{mode}_labels.pkl")
            if osp.exists(npz_path) and osp.exists(pkl_path):
                return npz_path, pkl_path
        return None

    def __getitem__(self, i):
        sample, label = self.data[i], self.label[i]
        if isinstance(sample, str):
            if not osp.exists(sample):
                raise FileNotFoundError(f"TieredImageNet image not found: {sample}")
            image = Image.open(sample).convert("RGB")
        else:
            image = Image.fromarray(sample.astype(np.uint8))
        if self.setname == "train":
            image = self.transform_train(image)
        else:
            image = self.transform_val_test(image)
        return image, label
