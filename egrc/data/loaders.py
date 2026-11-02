"""Original episodic loader builders with local package imports."""

from torch.utils.data import DataLoader
from .samplers import CategoriesSampler


def build_dataset(args):
    dataset_name = str(args.dataset).lower()
    if dataset_name == "miniimagenet":
        from egrc.data.mini_imagenet import MiniImageNet as Dataset

        return Dataset
    if dataset_name in {"tieredimagenet", "tiered-imagenet", "tiered_imagenet"}:
        from egrc.data.tiered_imagenet import TieredImagenet as Dataset

        return Dataset
    if dataset_name in {"cifarfs", "cifar-fs", "cifar_fs"}:
        from egrc.data.cifarfs import CIFARFS as Dataset

        return Dataset
    if dataset_name == "fc100":
        from egrc.data.fc100 import FC100 as Dataset

        return Dataset
    raise ValueError(f"Non-supported dataset: {args.dataset}")


def build_train_loaders(args, dataset_cls):
    print("Loading training split...", flush=True)
    trainset = dataset_cls("train", args)
    print(f"Training split: {len(trainset)} images, {trainset.num_class} classes.", flush=True)
    train_sampler = CategoriesSampler(
        trainset.label,
        args.train_episodes,
        args.way,
        args.shot + args.query,
        seed=getattr(args, "train_episode_seed", None),
        repeat_manifest=False,
    )
    train_loader = DataLoader(
        trainset, batch_sampler=train_sampler, num_workers=args.num_workers, pin_memory=True
    )
    print("Loading validation split...", flush=True)
    valset = dataset_cls("val", args)
    print(f"Validation split: {len(valset)} images, {valset.num_class} classes.", flush=True)
    val_sampler = CategoriesSampler(
        valset.label,
        args.val_episodes
        if getattr(args, "train_val_episodes", None) is None
        else int(args.train_val_episodes),
        args.test_way,
        args.shot + args.query,
        seed=getattr(args, "val_episode_seed", None)
        if getattr(args, "train_val_episode_seed", None) is None
        else int(args.train_val_episode_seed),
        repeat_manifest=bool(getattr(args, "train_val_repeat_manifest", True)),
    )
    val_loader = DataLoader(
        valset, batch_sampler=val_sampler, num_workers=args.num_workers, pin_memory=True
    )
    return (train_loader, val_loader)


def build_eval_loader(args, dataset_cls, split="test"):
    if split not in {"val", "test"}:
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    print(f"Loading {split} split...", flush=True)
    dataset = dataset_cls(split, args)
    print(
        f"{split.capitalize()} split: {len(dataset)} images, {dataset.num_class} classes.",
        flush=True,
    )
    episode_count = args.val_episodes if split == "val" else args.test_episodes
    sampler = CategoriesSampler(
        dataset.label,
        episode_count,
        args.test_way,
        args.shot + args.query,
        seed=getattr(args, f"{split}_episode_seed", None),
        repeat_manifest=True,
    )
    return DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=True)
