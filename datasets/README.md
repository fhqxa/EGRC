# Dataset layout

The dataset readers and preprocessing are included in `egrc/data/`.
Data files are supplied separately and ignored by Git.
The default root is this `datasets/` directory. You can instead use
`--dataset-root /path/to/your/existing/dataset`.

```text
datasets/
  mini-imagenet/
    images/                 Images, optionally grouped by class
    split/                  Alternative location for the split CSVs
    imagenet_class_index.json
    train.csv               filename,label header followed by samples
    val.csv
    test.csv
  tiered_imagenet/
    train/<class>/*.jpg
    val/<class>/*.jpg
    test/<class>/*.jpg
    class_names.json
  cifar_fs/
    data/<class>/*.png
    splits/
      train.txt
      val.txt
      test.txt
  FC100/
    train/<class>/*.png
    val/<class>/*.png
    test/<class>/*.png
```

Class directories can contain `.jpg`, `.jpeg`, `.png`, or `.bmp` files.
CIFAR-FS split text files list one class directory name per line. The reader
expects `train.txt`, `val.txt`, and `test.txt` directly under `splits/` or `split/`.
If an existing download places these files inside an additional named subfolder,
place the three class lists directly under `splits/` for this reader.
The JSON files shown above are optional metadata.

The original readers also accept these existing formats:

- miniImageNet: split CSVs under `split/`.
- tieredImageNet: `<split>_images.npz` plus `<split>_labels.pkl`.
- CIFAR-FS: `CIFAR_FS_<split>.pickle` (or `<split>.pickle`) with `data` and
  `labels`, or split text files listing class directories under `data/`.
- FC100: `meta-train/`, `meta-val/`, and `meta-test/` class directories.

CUB-200-2011 is a separate cross-domain dataset. Its directory structure is
documented in the [main README](../README.md#datasets); it is not used by the
main-experiment runner in this release.

FC100 uses the directory reader from the established main experiments; place
any archive-based FC100 data into that layout before training.
Keep the official class-disjoint train/validation/test splits. Every sampled
class needs at least `shot + query` images.

Preprocessing remains the original per-dataset 224x224 pipeline. Class sorting,
sample ordering, and shot-major episode ordering are retained. Support samples
occupy the first `way * shot` entries; query samples follow in the same class order.
