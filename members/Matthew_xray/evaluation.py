"""
members/Matthew_xray/evaluation.py — Step 6, fully standalone.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import config as cfg
from utils.visualization import plot_comparison_bar


def run():
    print(f"\n[{cfg.MEMBER_NAME}] Loading results...")
    ml = np.load(os.path.join(cfg.RESULTS_DIR, "ml_results.npy"), allow_pickle=True).item()
    dl = np.load(os.path.join(cfg.RESULTS_DIR, "dl_results.npy"), allow_pickle=True).item()

    def _bar_metrics(m):
        """Keys expected by plot_comparison_bar (weighted headline metrics)."""
        return {
            "accuracy": m["accuracy"],
            "precision": m.get("precision_weighted", m["precision"]),
            "recall": m.get("recall_weighted", m["recall"]),
            "f1": m.get("f1_weighted", m["f1"]),
            "train_time_sec": m.get("train_time_sec", 0),
            "precision_binary_pos": m.get("precision_binary_pos"),
            "recall_binary_pos": m.get("recall_binary_pos"),
            "f1_binary_pos": m.get("f1_binary_pos"),
            "f1_macro": m.get("f1_macro"),
        }

    models = {
        f"SVM ({cfg.MODEL_NAME} feats)": ml["svm"],
        f"LR  ({cfg.MODEL_NAME} feats)": ml["lr"],
        f"End-to-End {cfg.MODEL_NAME}":  dl,
    }
    models_bar = {k: _bar_metrics(v) for k, v in models.items()}

    print(f"\n{'='*72}")
    print(f"  {'Model':<30} {'Acc':>8} {'Prec*':>8} {'Rec*':>8} {'F1*':>8} {'Time(s)':>9}")
    print(f"  {'':30} {'':>8} {'(wtd)':>8} {'(wtd)':>8} {'(wtd)':>8} {'':>9}")
    print(f"{'='*72}")
    for name, m in models.items():
        mb = models_bar[name]
        print(f"  {name:<30} {mb['accuracy']:>8.4f} {mb['precision']:>8.4f} "
              f"{mb['recall']:>8.4f} {mb['f1']:>8.4f} {mb.get('train_time_sec',0):>9.1f}")
    print(f"{'='*72}")
    print("  *Prec/Rec/F1 = sklearn weighted averages (see README). "
          "Per-class detail: classification_report / confusion matrices.\n")

    plot_comparison_bar(models_bar,
        title=f"Approach 1 vs 2 — {cfg.DATASET_NAME} [{cfg.MODEL_NAME}] (weighted metrics)",
        save_path=os.path.join(cfg.RESULTS_DIR, "comparison_metrics.png"))

    # Training time chart
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    times  = [m.get("train_time_sec", 0) for m in models_bar.values()]
    bars   = ax.bar(list(models.keys()), times, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                f"{t:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Training Time (seconds)", fontsize=12)
    ax.set_title("Training Time Comparison", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_DIR, "comparison_time.png"), dpi=150)
    plt.show()
    print("[✓] Saved: comparison_time.png")

    # Confusion matrix collage
    cm_files = {"SVM": "cm_svm.png", "LR": "cm_lr.png", "End-to-End DL": "cm_dl.png"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Confusion Matrices — {cfg.MEMBER_NAME}", fontsize=14, fontweight="bold")
    for ax, (title, fname) in zip(axes, cm_files.items()):
        path = os.path.join(cfg.RESULTS_DIR, fname)
        if os.path.exists(path):
            ax.imshow(mpimg.imread(path)); ax.set_title(title, fontsize=11, fontweight="bold")
        else:
            ax.text(0.5, 0.5, f"Missing:\n{fname}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_DIR, "all_confusion_matrices.png"), dpi=150)
    plt.show()
    print("[✓] Saved: all_confusion_matrices.png")

    best = max(models_bar, key=lambda k: models_bar[k]["accuracy"])
    fast = min(models_bar, key=lambda k: models_bar[k].get("train_time_sec", 999))
    summary = (f"COMPARATIVE ANALYSIS — {cfg.MEMBER_NAME}\n"
               f"Dataset: {cfg.DATASET_NAME}  |  Model: {cfg.MODEL_NAME}\n"
               f"Table metrics: weighted Precision/Recall/F1 (see README).\n{'='*60}\n")
    for name, m in models_bar.items():
        summary += (f"{name:<32} Acc:{m['accuracy']:.4f}  "
                    f"F1_w:{m['f1']:.4f}  F1_binary_pos:{m.get('f1_binary_pos', float('nan')):.4f}  "
                    f"Time:{m.get('train_time_sec',0):.1f}s\n")
    summary += f"\nBest accuracy : {best}\nFastest       : {fast}\n"
    with open(os.path.join(cfg.RESULTS_DIR, "comparison_summary.txt"), "w") as f:
        f.write(summary)
    print(f"[✓] Saved: comparison_summary.txt\n[✓] Evaluation complete.\n")


if __name__ == "__main__":
    run()
