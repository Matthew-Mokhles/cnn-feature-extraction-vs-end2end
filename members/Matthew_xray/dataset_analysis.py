"""
members/Matthew_xray/dataset_analysis.py
────────────────────────────────────────────────────────────────────────────────
Step 1 — EDA for the Chest X-Ray dataset.
Run via:  python main.py --member matthew --step 1
"""

import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import config as cfg


def run():
    CLASS_NAMES = cfg.CLASS_NAMES
    RESULTS_DIR = cfg.RESULTS_DIR

    # ── Count images per split ─────────────────────────────────────────────
    def count_images(split_dir):
        counts = {}
        for cls in CLASS_NAMES:
            p = os.path.join(split_dir, cls)
            counts[cls] = len([f for f in os.listdir(p)
                                if f.lower().endswith((".jpg",".jpeg",".png"))]) if os.path.isdir(p) else 0
        return counts

    splits = {"Train": cfg.TRAIN_DIR, "Validation": cfg.VAL_DIR, "Test": cfg.TEST_DIR}

    print("\n" + "="*55)
    print(f"  {cfg.DATASET_NAME.upper()} — METADATA")
    print("="*55)
    print(f"{'Split':<15} {'NORMAL':>10} {'PNEUMONIA':>12} {'Total':>8}")
    print("-"*55)

    all_counts = {}
    grand_total = 0
    for name, d in splits.items():
        c = count_images(d)
        all_counts[name] = c
        total = sum(c.values())
        grand_total += total
        print(f"{name:<15} {c.get('NORMAL',0):>10} {c.get('PNEUMONIA',0):>12} {total:>8}")
    print("-"*55)
    print(f"{'TOTAL':<15} {'':>22} {grand_total:>8}\n")

    # ── Class distribution bar chart ───────────────────────────────────────
    colors = ["#4CAF50", "#F44336"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), constrained_layout=True)
    fig.suptitle("Class Distribution per Split", fontsize=14, fontweight="bold")
    for ax, (split_name, c) in zip(axes, all_counts.items()):
        bars = ax.bar(CLASS_NAMES, [c[cls] for cls in CLASS_NAMES], color=colors, width=0.5)
        ax.set_title(f"{split_name} Set", fontsize=12)
        ax.set_ylabel("Number of Images")
        ax.set_ylim(0, max(c.values()) * 1.25)
        for bar, val in zip(bars, [c[cls] for cls in CLASS_NAMES]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.savefig(os.path.join(RESULTS_DIR, "class_distribution.png"), dpi=150)
    plt.show()
    print(f"[✓] Saved: class_distribution.png")

    # ── Sample image grid ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6), constrained_layout=True)
    fig.suptitle("Sample Images — Training Set", fontsize=14, fontweight="bold")
    for col_idx, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(cfg.TRAIN_DIR, cls)
        files = random.sample([f for f in os.listdir(cls_dir)
                               if f.lower().endswith((".jpg",".jpeg",".png"))], 4)
        for row_idx, fname in enumerate(files):
            img = np.array(Image.open(os.path.join(cls_dir, fname)).convert("RGB").resize((224,224)))
            ax = fig.add_subplot(4, 2, row_idx * 2 + col_idx + 1)
            ax.imshow(img)
            ax.set_title(cls, fontsize=9, color=colors[col_idx], fontweight="bold")
            ax.axis("off")
    plt.savefig(os.path.join(RESULTS_DIR, "sample_images.png"), dpi=150)
    plt.show()
    print(f"[✓] Saved: sample_images.png")

    # ── Image size stats ───────────────────────────────────────────────────
    widths, heights = [], []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(cfg.TRAIN_DIR, cls)
        for f in os.listdir(cls_dir)[:100]:
            if f.lower().endswith((".jpg",".jpeg",".png")):
                with Image.open(os.path.join(cls_dir, f)) as img:
                    w, h = img.size
                    widths.append(w); heights.append(h)
    print(f"\n  Width  — min:{min(widths)} max:{max(widths)} mean:{np.mean(widths):.0f}")
    print(f"  Height — min:{min(heights)} max:{max(heights)} mean:{np.mean(heights):.0f}")
    print("\n[✓] Dataset analysis complete.\n")


if __name__ == "__main__":
    run()
