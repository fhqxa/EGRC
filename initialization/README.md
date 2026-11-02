# Pretrained ViT-S/16 weights

Use the same dataset-specific pretrained weights as the established main
experiments. Initialization files are not part of the GitHub source tree.

Download the **ViT-small** checkpoints from the
[Self-Supervised Pretrained Models section of FewTURE](https://github.com/mrkshllr/FewTURE#self-supervised-pretrained-models).
Use the 1600-epoch checkpoints for miniImageNet, CIFAR-FS, and FC100, and the
800-epoch checkpoint for tieredImageNet. Rename and place the downloaded files
as follows:

```text
initialization/
  miniimagenet/miniimagenet_checkpoint1600.pth
  tieredimagenet/tieredimagenet_checkpoint0800.pth
  cifar-fs/cifar-fs_checkpoint1600.pth
  fc100/fc100_checkpoint1600.pth
```

You can copy these files from your existing initialization directory, or pass
`--init-weights /path/to/the/corresponding/checkpoint.pth` for a single training.
These are pretrained backbone files, not trained EGRC checkpoints.

The loader uses the original teacher/backbone key mapping and checks at least
95% tensor coverage. Only load checkpoint files from a trusted source; the
original pretrained files contain metadata requiring standard PyTorch unpickling.
