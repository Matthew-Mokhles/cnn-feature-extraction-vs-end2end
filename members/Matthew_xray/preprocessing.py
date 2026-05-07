"""
members/Matthew_xray/preprocessing.py
────────────────────────────────────────────────────────────────────────────────
Step 2 — Data generators for the Chest X-Ray pipeline.

Enhancement: visualize_augmentation no longer hardcodes "PNEUMONIA" — it uses
cfg.CLASS_NAMES[-1] so it works if class names change.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config as cfg


def warn_if_validation_too_small(val_samples):
    """Print a prominent warning when val is too small for reliable DL callbacks."""
    if val_samples >= cfg.MIN_RECOMMENDED_VAL_SAMPLES:
        return
    bar = "!" * 72
    print("\n" + bar)
    print(f"[!] WARNING: Validation set has only {val_samples} images "
          f"(recommended >= {cfg.MIN_RECOMMENDED_VAL_SAMPLES}).")
    print("[!] val_accuracy / ModelCheckpoint / EarlyStopping are high-variance; "
          "trust TEST metrics for generalisation.")
    print("[!] Enlarge val/ (e.g. 10-20% stratified per class) or run "
          "`members/Matthew_xray/tools/split_train_val.py`.")
    print(bar + "\n")


def get_generators(batch_size=None, shuffle_test=False):
    """Returns (train_gen, val_gen, test_gen) using cfg settings."""
    bs = batch_size or cfg.BATCH_SIZE

    train_datagen = ImageDataGenerator(
        preprocessing_function=cfg.PREPROCESS_FN,
        **cfg.AUGMENTATION,
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=cfg.PREPROCESS_FN)

    common = dict(target_size=cfg.IMG_SIZE, batch_size=bs,
                  class_mode=cfg.CLASS_MODE, classes=cfg.CLASS_NAMES)

    train_gen = train_datagen.flow_from_directory(cfg.TRAIN_DIR, shuffle=True,  seed=42, **common)
    val_gen   = eval_datagen.flow_from_directory(cfg.VAL_DIR,   shuffle=False, **common)
    test_gen  = eval_datagen.flow_from_directory(cfg.TEST_DIR,  shuffle=shuffle_test, **common)
    return train_gen, val_gen, test_gen


def visualize_augmentation(n_images=8):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    # FIX: no longer hardcodes "PNEUMONIA" — uses last class in CLASS_NAMES
    sample_cls = cfg.CLASS_NAMES[-1]
    cls_dir    = os.path.join(cfg.TRAIN_DIR, sample_cls)
    all_files  = [f for f in os.listdir(cls_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files      = all_files[:n_images]   # safe — no crash if fewer than n_images

    aug_gen = ImageDataGenerator(**cfg.AUGMENTATION)
    n = len(files)
    if n == 0:
        print("[!] No images found for augmentation visualisation.")
        return

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4), constrained_layout=True)
    fig.suptitle(f"Top: Original  |  Bottom: Augmented  ({sample_cls})",
                 fontsize=12, fontweight="bold")

    for i, fname in enumerate(files):
        arr = img_to_array(load_img(os.path.join(cls_dir, fname),
                                    target_size=cfg.IMG_SIZE))
        col = axes[:, i] if n > 1 else axes
        col[0].imshow(arr.astype("uint8")); col[0].axis("off")
        aug = next(aug_gen.flow(arr[np.newaxis], batch_size=1))[0]
        col[1].imshow(aug.astype("uint8"));  col[1].axis("off")

    plt.savefig(os.path.join(cfg.RESULTS_DIR, "augmentation_samples.png"), dpi=150)
    plt.show()
    print("[✓] Saved: augmentation_samples.png")


def run():
    print("\n[*] Building data generators...")
    train_gen, val_gen, test_gen = get_generators()
    warn_if_validation_too_small(val_gen.samples)
    print(f"  Train  — {train_gen.samples} images | {cfg.NUM_CLASSES} classes")
    print(f"  Val    — {val_gen.samples} images")
    print(f"  Test   — {test_gen.samples} images")
    x, y = next(train_gen)
    print(f"  Batch  — shape:{x.shape}  dtype:{x.dtype}  labels:{y[:6]}")
    print(f"  Augmentation: {list(cfg.AUGMENTATION.keys())}")
    print("\n[*] Visualising augmented samples...")
    visualize_augmentation()
    print("\n[✓] Preprocessing ready.\n")


if __name__ == "__main__":
    run()
