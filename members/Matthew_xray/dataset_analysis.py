"""
members/Matthew_xray/dataset_analysis.py
────────────────────────────────────────────────────────────────────────────────
Step 1 — EDA for the Chest X-Ray dataset.

Enhancements over original:
  - Safe sample grid: never crashes when a class has fewer than 4 images
    (Kaggle val/ has only 8 images total).
  - Imbalance ratio warning: flags when PNEUMONIA:NORMAL > 2:1.
  - Grayscale detection: reports whether images are true grayscale or
    RGB-wrapped grayscale (chest x-rays are almost always the latter).
  - Pixel intensity histogram: side-by-side comparison of NORMAL vs
    PNEUMONIA distributions on a random sample — helps spot CLAHE benefit.
  - Image size scatter plot: reveals resolution variability before resizing.
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import config as cfg


# ── Helpers ───────────────────────────────────────────────────────────────────

def _list_images(folder):
    if not os.path.isdir(folder):
        return []
    return [f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]


def _count_images(split_dir):
    return {cls: len(_list_images(os.path.join(split_dir, cls)))
            for cls in cfg.CLASS_NAMES}


def _safe_sample(files, n):
    """Return up to n items — never crashes when len(files) < n."""
    return random.sample(files, min(n, len(files)))


def _load_rgb(path):
    return np.array(Image.open(path).convert("RGB").resize(cfg.IMG_SIZE))


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    CLASS_NAMES = cfg.CLASS_NAMES
    RESULTS_DIR = cfg.RESULTS_DIR

    # ── 1. Count & print metadata ──────────────────────────────────────────
    splits = {"Train": cfg.TRAIN_DIR, "Validation": cfg.VAL_DIR, "Test": cfg.TEST_DIR}
    all_counts = {}
    grand_total = 0

    print("\n" + "=" * 60)
    print(f"  {cfg.DATASET_NAME.upper()} — DATASET METADATA")
    print("=" * 60)
    print(f"{'Split':<15} {CLASS_NAMES[0]:>10} {CLASS_NAMES[1]:>12} {'Total':>8}  {'Imbal':>7}")
    print("-" * 60)

    for name, d in splits.items():
        c = _count_images(d)
        all_counts[name] = c
        total = sum(c.values())
        grand_total += total
        n0, n1 = c.get(CLASS_NAMES[0], 0), c.get(CLASS_NAMES[1], 0)
        ratio = f"{n1/max(n0,1):.1f}:1" if n0 > 0 else "N/A"
        flag  = " ⚠" if n1 > 2 * n0 else ""
        print(f"{name:<15} {n0:>10} {n1:>12} {total:>8}  {ratio:>7}{flag}")

    print("-" * 60)
    print(f"{'TOTAL':<15} {'':>22} {grand_total:>8}\n")

    train_c = all_counts["Train"]
    n0_tr, n1_tr = train_c.get(CLASS_NAMES[0], 0), train_c.get(CLASS_NAMES[1], 0)
    if n1_tr > 2 * n0_tr:
        print(f"  [⚠] Class imbalance detected in training set: "
              f"{CLASS_NAMES[1]}/{CLASS_NAMES[0]} = {n1_tr/max(n0_tr,1):.1f}:1")
        print(f"      → class_weight='balanced' is active in endtoend_dl.py.\n")

    # ── 2. Class distribution bar chart ───────────────────────────────────
    colors = ["#4CAF50", "#F44336"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), constrained_layout=True)
    fig.suptitle("Class Distribution per Split", fontsize=14, fontweight="bold")
    for ax, (split_name, c) in zip(axes, all_counts.items()):
        vals = [c.get(cls, 0) for cls in CLASS_NAMES]
        bars = ax.bar(CLASS_NAMES, vals, color=colors, width=0.5)
        ax.set_title(f"{split_name} Set", fontsize=12)
        ax.set_ylabel("Number of Images")
        ax.set_ylim(0, max(vals) * 1.28 if max(vals) > 0 else 10)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    str(val), ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
    plt.savefig(os.path.join(RESULTS_DIR, "class_distribution.png"), dpi=150)
    plt.show()
    print("[✓] Saved: class_distribution.png")

    # ── 3. Safe sample image grid ──────────────────────────────────────────
    n_cols  = len(CLASS_NAMES)
    n_rows  = 4
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 3.5 * n_rows),
                             constrained_layout=True)
    fig.suptitle("Sample Images — Training Set", fontsize=14, fontweight="bold")

    for col_idx, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(cfg.TRAIN_DIR, cls)
        files   = _safe_sample(_list_images(cls_dir), n_rows)   # FIX: safe sample
        for row_idx in range(n_rows):
            ax = axes[row_idx, col_idx]
            if row_idx < len(files):
                img = _load_rgb(os.path.join(cls_dir, files[row_idx]))
                ax.imshow(img, cmap="gray")
                ax.set_title(cls, fontsize=9,
                             color=colors[col_idx], fontweight="bold")
            else:
                ax.text(0.5, 0.5, "no image", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")
            ax.axis("off")

    plt.savefig(os.path.join(RESULTS_DIR, "sample_images.png"), dpi=150)
    plt.show()
    print("[✓] Saved: sample_images.png")

    # ── 4. Grayscale detection ─────────────────────────────────────────────
    print("\n[*] Checking grayscale vs RGB-wrapped...")
    cls_dir = os.path.join(cfg.TRAIN_DIR, CLASS_NAMES[0])
    probe   = _safe_sample(_list_images(cls_dir), 10)
    true_gray = 0
    for fname in probe:
        arr = np.array(Image.open(os.path.join(cls_dir, fname)).convert("RGB"))
        if np.allclose(arr[:, :, 0], arr[:, :, 1], atol=2) and \
           np.allclose(arr[:, :, 1], arr[:, :, 2], atol=2):
            true_gray += 1
    pct = true_gray / max(len(probe), 1) * 100
    print(f"  {true_gray}/{len(probe)} probed images are effectively grayscale ({pct:.0f}%).")
    if pct > 50:
        print("  → Images are RGB-wrapped grayscale (3 identical channels).")
        print("    CLAHE operates on the L channel (LAB) — effective for this data.\n")
    else:
        print("  → Images appear to be true colour RGB.\n")

    # ── 5. Pixel intensity histogram ───────────────────────────────────────
    print("[*] Building pixel intensity histograms (sample of 50 images/class)...")
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for cls, color in zip(CLASS_NAMES, colors):
        cls_dir = os.path.join(cfg.TRAIN_DIR, cls)
        files   = _safe_sample(_list_images(cls_dir), 50)
        pixels  = []
        for fname in files:
            img = np.array(Image.open(os.path.join(cls_dir, fname))
                           .convert("L").resize(cfg.IMG_SIZE), dtype=np.float32)
            pixels.append(img.ravel())
        all_px = np.concatenate(pixels)
        ax.hist(all_px, bins=128, range=(0, 255), density=True,
                alpha=0.55, color=color, label=cls)

    ax.set_xlabel("Pixel Intensity (grayscale)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Pixel Intensity Distribution — NORMAL vs PNEUMONIA\n"
                 "(before preprocessing; shows contrast difference CLAHE targets)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    plt.savefig(os.path.join(RESULTS_DIR, "intensity_histogram.png"), dpi=150)
    plt.show()
    print("[✓] Saved: intensity_histogram.png")

    # ── 6. Image size scatter ──────────────────────────────────────────────
    print("[*] Sampling image resolutions from training set (first 100/class)...")
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for cls, color, marker in zip(CLASS_NAMES, colors, ["o", "^"]):
        cls_dir = os.path.join(cfg.TRAIN_DIR, cls)
        files   = _list_images(cls_dir)[:100]
        ws, hs  = [], []
        for fname in files:
            with Image.open(os.path.join(cls_dir, fname)) as im:
                ws.append(im.width); hs.append(im.height)
        ax.scatter(ws, hs, alpha=0.4, s=20, color=color,
                   marker=marker, label=cls)
        print(f"  {cls:<12} W: {min(ws):>4}–{max(ws):<4}  "
              f"H: {min(hs):>4}–{max(hs):<4}  mean: {np.mean(ws):.0f}×{np.mean(hs):.0f}")

    ax.axvline(cfg.IMG_SIZE[1], color="navy",  ls="--", lw=1.2, label=f"Target W={cfg.IMG_SIZE[1]}")
    ax.axhline(cfg.IMG_SIZE[0], color="black", ls="--", lw=1.2, label=f"Target H={cfg.IMG_SIZE[0]}")
    ax.set_xlabel("Width (px)", fontsize=11); ax.set_ylabel("Height (px)", fontsize=11)
    ax.set_title("Native Image Resolutions — Training Set\n"
                 "(dashed lines = resize target 224×224)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(linestyle="--", alpha=0.4)
    plt.savefig(os.path.join(RESULTS_DIR, "resolution_scatter.png"), dpi=150)
    plt.show()
    print("[✓] Saved: resolution_scatter.png")

    print("\n[✓] Dataset analysis complete.\n")


if __name__ == "__main__":
    run()
