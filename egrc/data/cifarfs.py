import os
import os.path as osp
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .dataset_readers import list_image_files, resolve_dataset_root


DEFAULT_ROOT = "datasets"


class CIFARFS(Dataset):
    def __init__(self, setname, args):
        configured_root = getattr(args, "dataset_root", None) or DEFAULT_ROOT
        self.root = resolve_dataset_root(
            configured_root,
            ["cifar_fs", "cifar-fs", "cifarfs", "CIFAR-FS", "CIFAR_FS"],
            validator=self._is_valid_root,
        )
        class_dict = self._load_class_dict(self.root, setname)
        if not class_dict:
            raise RuntimeError(f"No valid classes found for CIFAR-FS {setname} split.")

        self.class_dict = class_dict
        self.classes = sorted(class_dict.keys())
        self.data = []
        self.label = []
        for idx, class_name in enumerate(self.classes):
            for sample in class_dict[class_name]:
                self.data.append(sample)
                self.label.append(idx)
        self.num_class = len(self.classes)
        self.setname = setname

        image_size = 224
        # Match the CIFAR-FS preprocessing used by FewTURE/DeepEMD.
        mean = np.array([0.5071, 0.4866, 0.4409])
        std = np.array([0.2009, 0.1984, 0.2023])

        self.transform_train = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(value=[0.5071, 0.4866, 0.4409]),
            ]
        )
        self.transform_val_test = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _is_valid_root(dataset_root):
        for mode in ("train", "val", "test"):
            if CIFARFS._find_pickle_path(dataset_root, mode) is not None:
                return True
            if CIFARFS._find_split_file(dataset_root, mode) is not None:
                return True
            if CIFARFS._find_folder_split(dataset_root, mode) is not None:
                return True
        return False

    @staticmethod
    def _load_class_dict(dataset_root, mode):
        class_dict = {}

        folder_root = CIFARFS._find_folder_split(dataset_root, mode)
        if folder_root is not None:
            for class_name in sorted(os.listdir(folder_root)):
                class_dir = osp.join(folder_root, class_name)
                if not osp.isdir(class_dir):
                    continue
                for image_file in list_image_files(class_dir):
                    class_dict.setdefault(class_name, []).append(osp.join(class_dir, image_file))
            if class_dict:
                return class_dict

        split_file = CIFARFS._find_split_file(dataset_root, mode)
        if split_file is not None:
            split_root = osp.dirname(osp.dirname(split_file))
            data_dir_candidates = [
                osp.join(split_root, "data"),
                split_root,
            ]
            with open(split_file, "r", encoding="utf-8") as handle:
                classes = [line.strip() for line in handle if line.strip()]
            for class_name in classes:
                class_dir = None
                for data_dir in data_dir_candidates:
                    candidate = osp.join(data_dir, class_name)
                    if osp.isdir(candidate):
                        class_dir = candidate
                        break
                if class_dir is None:
                    continue
                for image_file in list_image_files(class_dir):
                    class_dict.setdefault(class_name, []).append(osp.join(class_dir, image_file))
            if class_dict:
                return class_dict

        pickle_path = CIFARFS._find_pickle_path(dataset_root, mode)
        if pickle_path is not None:
            with open(pickle_path, "rb") as handle:
                unpickler = pickle._Unpickler(handle)
                unpickler.encoding = "latin1"
                data = unpickler.load()
            for image, label in zip(data["data"], data["labels"]):
                class_dict.setdefault(int(label), []).append(image)
            return class_dict

        raise FileNotFoundError(
            f"CIFAR-FS split not found under {dataset_root}. "
            f"Checked pickle, split txt, and folder layouts for {mode}."
        )

    @staticmethod
    def _find_pickle_path(dataset_root, mode):
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
            for candidate in [
                osp.join(root, f"CIFAR_FS_{mode}.pickle"),
                osp.join(root, f"cifar_fs_{mode}.pickle"),
                osp.join(root, f"{mode}.pickle"),
            ]:
                if osp.exists(candidate):
                    return candidate
        return None

    @staticmethod
    def _find_split_file(dataset_root, mode):
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
            for candidate in [
                osp.join(root, "splits", f"{mode}.txt"),
                osp.join(root, "split", f"{mode}.txt"),
            ]:
                if osp.exists(candidate):
                    return candidate
        return None

    @staticmethod
    def _find_folder_split(dataset_root, mode):
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
            for candidate in [
                osp.join(root, mode),
                osp.join(root, f"meta-{mode}"),
                osp.join(root, "data", mode),
            ]:
                if osp.isdir(candidate):
                    return candidate
        return None

    def __getitem__(self, i):
        sample, label = self.data[i], self.label[i]
        if isinstance(sample, str):
            if not osp.exists(sample):
                raise FileNotFoundError(f"CIFAR-FS image not found: {sample}")
            image = Image.open(sample).convert("RGB")
        else:
            image = Image.fromarray(sample.astype(np.uint8))
        if self.setname == "train":
            image = self.transform_train(image)
        else:
            image = self.transform_val_test(image)
        return image, label
