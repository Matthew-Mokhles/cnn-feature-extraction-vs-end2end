"""
utils/visualization.py — Matthew's local copy, no shared dependencies.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


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
