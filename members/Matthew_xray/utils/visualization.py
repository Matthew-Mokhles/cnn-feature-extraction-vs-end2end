"""
utils/visualization.py — Matthew's local copy, no shared dependencies.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score,
)
from sklearn.calibration import calibration_curve


def plot_confusion_matrix(y_true, y_pred, class_names,
                          title="Confusion Matrix", save_path=None, figsize=(6, 5)):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_norm, annot=False, cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor="white")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j+0.5, i+0.5,
                    f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if cm_norm[i,j] > 0.5 else "black")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title(title,              fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  [✓] Saved: {os.path.basename(save_path)}")
    plt.show()
    return fig


def plot_training_curves(history, title="Training History",
                         save_path=None, figsize=(13, 5)):
    epochs = range(1, len(history["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    ax1.plot(epochs, history["accuracy"],     "b-o", ms=4, label="Train")
    ax1.plot(epochs, history["val_accuracy"], "r-o", ms=4, label="Val")
    ax1.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.05))
    ax1.legend(); ax1.grid(linestyle="--", alpha=0.5)
    ax2.plot(epochs, history["loss"],     "b-o", ms=4, label="Train")
    ax2.plot(epochs, history["val_loss"], "r-o", ms=4, label="Val")
    ax2.set(title="Loss", xlabel="Epoch", ylabel="Loss")
    ax2.legend(); ax2.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  [✓] Saved: {os.path.basename(save_path)}")
    plt.show()
    return fig


def plot_comparison_bar(models_dict, title="Model Comparison",
                        save_path=None, figsize=(12, 6)):
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    n_models  = len(models_dict)
    bar_w = 0.8 / n_models
    x     = np.arange(len(metrics))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]
    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, m) in enumerate(models_dict.items()):
        vals   = [m.get(k, 0) for k in metrics]
        offset = (i - n_models / 2 + 0.5) * bar_w
        bars   = ax.bar(x + offset, vals, bar_w, label=name,
                        color=colors[i % len(colors)], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  [✓] Saved: {os.path.basename(save_path)}")
    plt.show()
    return fig


# ── Probability-aware diagnostics ─────────────────────────────────────────────

def plot_roc(curves, title="ROC Curves", save_path=None, figsize=(7, 6)):
    """``curves`` is a dict of {model_name: (y_true, y_proba)}."""
    fig, ax = plt.subplots(figsize=figsize)
    for name, (y_true, y_proba) in curves.items():
        y_true = np.asarray(y_true).astype(int).ravel()
        y_proba = np.asarray(y_proba).astype(float).ravel()
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4); ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  [✓] Saved: {os.path.basename(save_path)}")
    plt.show()
    return fig


def plot_pr(curves, title="Precision–Recall Curves", save_path=None, figsize=(7, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    for name, (y_true, y_proba) in curves.items():
        y_true = np.asarray(y_true).astype(int).ravel()
        y_proba = np.asarray(y_proba).astype(float).ravel()
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        ax.plot(rec, prec, lw=2, label=f"{name} (AP = {ap:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall", fontsize=11); ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4); ax.legend(fontsize=9, loc="lower left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  [✓] Saved: {os.path.basename(save_path)}")
    plt.show()
    return fig


def plot_calibration(y_true, y_proba, n_bins=10, title="Calibration",
                     save_path=None, figsize=(6, 6)):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = np.asarray(y_proba).astype(float).ravel()
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins,
                                            strategy="quantile")
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", lw=2, label="Model")
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of positives",     fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(linestyle="--", alpha=0.4); ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  [✓] Saved: {os.path.basename(save_path)}")
    plt.show()
    return fig


def plot_threshold_sweep(y_true, y_proba, pos_label=1,
                         title="Threshold Sweep", save_path=None,
                         figsize=(8, 5), highlight=None):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = np.asarray(y_proba).astype(float).ravel()
    grid = np.linspace(0.05, 0.95, 91)
    f1s, precs, recs = [], [], []
    for t in grid:
        y_pred = (y_proba >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, average="binary",
                            pos_label=pos_label, zero_division=0))
        precs.append(precision_score(y_true, y_pred, average="binary",
                                     pos_label=pos_label, zero_division=0))
        recs.append(recall_score(y_true, y_pred, average="binary",
                                 pos_label=pos_label, zero_division=0))
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(grid, f1s,   lw=2, label="F1")
    ax.plot(grid, precs, lw=2, label="Precision")
    ax.plot(grid, recs,  lw=2, label="Recall")
    if highlight is not None:
        ax.axvline(highlight, color="red", ls="--", lw=1,
                   label=f"chosen = {highlight:.2f}")
    ax.set_xlabel("Decision threshold", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.02); ax.grid(linestyle="--", alpha=0.4); ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  [✓] Saved: {os.path.basename(save_path)}")
    plt.show()
    return fig
