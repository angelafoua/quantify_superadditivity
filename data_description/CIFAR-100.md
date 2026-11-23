# CIFAR-100 Dataset Description

This document describes the CIFAR-100 dataset used as the primary dataset in this project: what it contains, how the files are stored, how images and labels are encoded, and what the images look like.

Reference: Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images.* University of Toronto technical report. Official page: https://www.cs.toronto.edu/~kriz/cifar.html

---

## 1. Overview

| Property | Value |
|----------|-------|
| Full name | CIFAR-100 (Canadian Institute for Advanced Research, 100 classes) |
| Task | Image classification |
| Total images | 60,000 |
| Training images | 50,000 |
| Test images | 10,000 |
| Fine classes | 100 |
| Coarse classes (superclasses) | 20 |
| Images per fine class | 600 (500 train, 100 test) |
| Image size | 32 × 32 pixels |
| Channels | 3 (RGB, color) |
| Pixel depth | 8 bits per channel (`uint8`, values 0–255) |
| Raw size per image | 3,072 bytes (32 × 32 × 3) |
| Archive | `cifar-100-python.tar.gz`, about 161 MiB |

The official train/test split is fixed by the dataset authors and the two sets do not overlap. The project's probe set (fixed seed `999`) is a separate construction and does not change this split.

---

## 2. Files in the archive

The Python version extracts to `data/cifar-100-python/`, which contains three files. They are Python pickle files, not text.

| File | Contents | Approx. size |
|------|----------|--------------|
| `train` | 50,000 training images and labels | ~150 MB |
| `test` | 10,000 test images and labels | ~30 MB |
| `meta` | Class names (fine and coarse) | < 1 KB |

Each `train` and `test` file is a pickled `dict` with byte-string keys:

| Key | Type | Shape / value |
|-----|------|---------------|
| `b'data'` | `numpy.ndarray`, `uint8` | `(N, 3072)`, N = 50,000 (train) or 10,000 (test) |
| `b'fine_labels'` | `list[int]` | length N, values 0–99 |
| `b'coarse_labels'` | `list[int]` | length N, values 0–19 |
| `b'filenames'` | `list[bytes]` | original image filenames |
| `b'batch_label'` | `bytes` | e.g. `b'training batch 1 of 1'` |

The `meta` file is a dict with:

| Key | Contents |
|-----|----------|
| `b'fine_label_names'` | list of 100 class names (index = fine label) |
| `b'coarse_label_names'` | list of 20 superclass names (index = coarse label) |

**Loading:** use `pickle.load(f, encoding="bytes")`. Unpickling can execute arbitrary code, so load only verified files (see Section 7).

---

## 3. Image encoding

Each row of `data` is one image, stored in planar channel order:

```
row of 3072 bytes = [ R channel: 1024 values | G channel: 1024 values | B channel: 1024 values ]
each channel      = 32 rows × 32 columns, row-major
```

To get a standard `(height, width, channel)` image:

```python
img = data[i].reshape(3, 32, 32).transpose(1, 2, 0)   # -> (32, 32, 3), uint8
```

Colors are standard 8-bit RGB: 0 is black, 255 is full intensity. There is no alpha channel.

Commonly used per-channel normalization statistics for the training set (on the [0, 1] scale):

| Channel | Mean | Std |
|---------|------|-----|
| R | 0.5071 | 0.2673 |
| G | 0.4865 | 0.2564 |
| B | 0.4409 | 0.2762 |

These are widely used values. Recompute them from the project's own training split if exact reproducibility matters.

---

## 4. Labels

### 4.1 Fine labels (100 classes)

Integer labels 0–99. The label is the index into `fine_label_names` in `meta`. The order is alphabetical:

| Idx | Class | Idx | Class | Idx | Class | Idx | Class | Idx | Class |
|-----|-------|-----|-------|-----|-------|-----|-------|-----|-------|
| 0 | apple | 20 | chair | 40 | lamp | 60 | plain | 80 | squirrel |
| 1 | aquarium_fish | 21 | chimpanzee | 41 | lawn_mower | 61 | plate | 81 | streetcar |
| 2 | baby | 22 | clock | 42 | leopard | 62 | poppy | 82 | sunflower |
| 3 | bear | 23 | cloud | 43 | lion | 63 | porcupine | 83 | sweet_pepper |
| 4 | beaver | 24 | cockroach | 44 | lizard | 64 | possum | 84 | table |
| 5 | bed | 25 | couch | 45 | lobster | 65 | rabbit | 85 | tank |
| 6 | bee | 26 | crab | 46 | man | 66 | raccoon | 86 | telephone |
| 7 | beetle | 27 | crocodile | 47 | maple_tree | 67 | ray | 87 | television |
| 8 | bicycle | 28 | cup | 48 | motorcycle | 68 | road | 88 | tiger |
| 9 | bottle | 29 | dinosaur | 49 | mountain | 69 | rocket | 89 | tractor |
| 10 | bowl | 30 | dolphin | 50 | mouse | 70 | rose | 90 | train |
| 11 | boy | 31 | elephant | 51 | mushroom | 71 | sea | 91 | trout |
| 12 | bridge | 32 | flatfish | 52 | oak_tree | 72 | seal | 92 | tulip |
| 13 | bus | 33 | forest | 53 | orange | 73 | shark | 93 | turtle |
| 14 | butterfly | 34 | fox | 54 | orchid | 74 | shrew | 94 | wardrobe |
| 15 | camel | 35 | girl | 55 | otter | 75 | skunk | 95 | whale |
| 16 | can | 36 | hamster | 56 | palm_tree | 76 | skyscraper | 96 | willow_tree |
| 17 | castle | 37 | house | 57 | pear | 77 | snail | 97 | wolf |
| 18 | caterpillar | 38 | kangaroo | 58 | pickup_truck | 78 | snake | 98 | woman |
| 19 | cattle | 39 | keyboard | 59 | pine_tree | 79 | spider | 99 | worm |

This table was written from the published CIFAR-100 class list. It has not been checked against the downloaded files. The authoritative source is `meta[b'fine_label_names']`. Read names from `meta` at runtime rather than hard-coding this table.

The classes cover animals, fish, flowers, food containers, fruit and vegetables, household electrical devices and furniture, insects, large carnivores, large outdoor man-made and natural scenes, large omnivores and herbivores, medium and small mammals, non-insect invertebrates, people, reptiles, trees, and vehicles.

### 4.2 Coarse labels (20 superclasses)

Each fine class belongs to exactly one superclass, so `coarse_labels` is determined by `fine_labels`. In index order:

`0` aquatic_mammals · `1` fish · `2` flowers · `3` food_containers · `4` fruit_and_vegetables · `5` household_electrical_devices · `6` household_furniture · `7` insects · `8` large_carnivores · `9` large_man-made_outdoor_things · `10` large_natural_outdoor_scenes · `11` large_omnivores_and_herbivores · `12` medium_mammals · `13` non-insect_invertebrates · `14` people · `15` reptiles · `16` small_mammals · `17` trees · `18` vehicles_1 · `19` vehicles_2

Each superclass has 5 fine classes. For example, `people` contains baby, boy, girl, man, and woman.

### 4.3 Label properties

- Labels are integers, not one-hot vectors.
- Classes are balanced: each fine class has exactly 500 training and 100 test images.

---

## 5. Image content

- Images are 32 × 32 thumbnails. At this size, fine-grained classes (for example, similar tree species or vehicle types) are hard to tell apart, and small objects lose detail.
- Objects are usually centered and fill much of the frame, but pose, lighting, background, and scale vary.
- Each image is labeled with one fine class, but backgrounds can contain other objects.
- The stored images are not augmented. Random crops, flips, and similar transforms should be applied at training time.

---

## 6. Relevance to this project

- Used as the primary dataset for the factorial experiment (IID / non-IID × dense / community topology).
- Non-IID partitioning is applied to the training split. It must not change the official train/test split.
- The test split is for evaluation only. It must not be used for partitioning, probe selection, or hyperparameter tuning.
- The probe set uses fixed seed `999` and is identical across all runs, as required by the invariants in `CLAUDE.md`.

---

## 7. Security and integrity

Layer-risk view:

| Layer | Risk | Safeguard |
|-------|------|-----------|
| Source / supply chain | Tampered or substituted archive | Download only from the official Toronto URL. Record the archive's SHA-256 and verify it before extraction. |
| Data at rest | Pickle files execute code when loaded | Load only verified archives. Run loaders with `python -I`. Do not load pickles from unknown sources. |
| Data at rest | ~160 MB dataset committed to git | Keep `data/` out of git. Check with `git check-ignore -v data/cifar-100-python.tar.gz`. |
| Integrity | Partial download or corruption | List contents with `tar -tzf`, then check array shapes (see Section 8). |

**Priority:** the highest-impact protection is verifying the archive checksum and never unpickling files from an unverified source.

---

## 8. Quick verification

```python
import pickle
import numpy as np

with open("data/cifar-100-python/train", "rb") as f:
    d = pickle.load(f, encoding="bytes")
with open("data/cifar-100-python/meta", "rb") as f:
    meta = pickle.load(f, encoding="bytes")

assert d[b"data"].shape == (50000, 3072) and d[b"data"].dtype == np.uint8
assert len(meta[b"fine_label_names"]) == 100
assert len(meta[b"coarse_label_names"]) == 20
assert np.bincount(d[b"fine_labels"]).min() == 500
print("CIFAR-100 train split OK")
```
