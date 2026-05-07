"""
members/Matthew_xray/evaluation.py — Step 6, fully standalone.

Reads probabilities saved by Step 4 (ML) and Step 5 (DL), tunes the decision
threshold on the **val** split (F1 on PNEUMONIA), and reports test metrics at
both the default 0.5 threshold and the tuned one. Emits ROC, PR, threshold
sweep, and calibration plots in addition to the legacy comparison artifacts.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import config as cfg
from utils.metrics import find_best_threshold, compute_metrics
from utils.visualization import (
    plot_comparison_bar, plot_roc, plot_pr, plot_calibration,
    plot_threshold_sweep, plot_confusion_matrix,
)


def _load_proba(name):
    """Return (y_val, p_val, y_test, p_test) or None if any file is missing."""
    val_path  = os.path.join(cfg.RESULTS_DIR, f"{name}_val_proba.npy")
    test_path = os.path.join(cfg.RESULTS_DIR, f"{name}_test_proba.npy")
    if not (os.path.exists(val_path) and os.path.exists(test_path)):
        return None
    val_arr  = np.load(val_path)
    test_arr = np.load(test_path)
    return (val_arr[:, 0].astype(int), val_arr[:, 1].astype(float),
            test_arr[:, 0].astype(int), test_arr[:, 1].astype(float))


def _bar_metrics(m):
    return {
        "accuracy":  m.get("accuracy"),
        "precision": m.get("precision_weighted", m.get("precision")),
        "recall":    m.get("recall_weighted",    m.get("recall")),
        "f1":        m.get("f1_weighted",        m.get("f1")),
        "train_time_sec":       m.get("train_time_sec", 0),
        "precision_binary_pos": m.get("precision_binary_pos"),
        "recall_binary_pos":    m.get("recall_binary_pos"),
        "f1_binary_pos":        m.get("f1_binary_pos"),
        "f1_macro":             m.get("f1_macro"),
        "roc_auc":              m.get("roc_auc"),
        "pr_auc":               m.get("pr_auc"),
        "tuned_threshold":      m.get("tuned_threshold"),
        "tuned_accuracy":       m.get("tuned_accuracy"),
        "tuned_f1_weighted":    m.get("tuned_f1_weighted"),
        "tuned_f1_binary_pos":  m.get("tuned_f1_binary_pos"),
    }


def _retune_on_val(model_key, label, base_metrics):
    """Tune threshold on val, evaluate on test, and return enriched metrics."""
    proba = _load_proba(model_key)
    if proba is None:
        print(f"[!] No probabilities for {label}; using default-threshold metrics only.")
        return base_metrics, None
    y_val, p_val, y_test, p_test = proba

    t_star, _ = find_best_threshold(y_val, p_val, pos_label=1, criterion="f1")
    y_pred_default = (p_test >= 0.5).astype(int)
    y_pred_tuned   = (p_test >= t_star).astype(int)

    print(f"\n[{label}] threshold tuned on val → {t_star:.2f}")
    m_default = compute_metrics(y_test, y_pred_default,
                                model_name=f"{label} @ 0.50")
    m_tuned   = compute_metrics(y_test, y_pred_tuned,
                                model_name=f"{label} @ {t_star:.2f} (tuned on val)")

    enriched = dict(base_metrics)
    enriched["tuned_threshold"] = float(t_star)
    for k, v in m_tuned.items():
        enriched[f"tuned_{k}"] = v
    if "roc_auc" not in enriched or enriched.get("roc_auc") is None:
        from sklearn.metrics import roc_auc_score, average_precision_score
        try: enriched["roc_auc"] = float(roc_auc_score(y_test, p_test))
        except Exception: pass
        try: enriched["pr_auc"]  = float(average_precision_score(y_test, p_test))
        except Exception: pass
    return enriched, (y_test, p_test, t_star)


def run():
    print(f"\n[{cfg.MEMBER_NAME}] Loading results...")
    ml = np.load(os.path.join(cfg.RESULTS_DIR, "ml_results.npy"), allow_pickle=True).item()
    dl = np.load(os.path.join(cfg.RESULTS_DIR, "dl_results.npy"), allow_pickle=True).item()

    raw_models = {
        f"SVM ({cfg.MODEL_NAME} feats)": ("svm", ml.get("svm")),
        f"LR  ({cfg.MODEL_NAME} feats)": ("lr",  ml.get("lr")),
    }
    if "mlp" in ml:
        raw_models[f"MLP ({cfg.MODEL_NAME} feats)"] = ("mlp", ml["mlp"])
    raw_models[f"End-to-End {cfg.MODEL_NAME}"] = ("dl", dl)

    enriched_models = {}
    proba_pairs = {}  # for ROC / PR / calibration plots
    for label, (key, m) in raw_models.items():
        if m is None:
            continue
        enriched, proba = _retune_on_val(key, label, m)
        enriched_models[label] = enriched
        if proba is not None:
            y_test, p_test, _ = proba
            proba_pairs[label] = (y_test, p_test)

    models_bar = {k: _bar_metrics(v) for k, v in enriched_models.items()}

    # ── Headline table ───────────────────────────────────────────────
    print(f"\n{'='*92}")
    print(f"  {'Model':<32} {'Acc':>8} {'F1*':>8} {'AUC':>6} {'PR-AUC':>7} "
          f"{'TunedF1b':>9} {'Time(s)':>9}")
    print(f"  {'':32} {'':>8} {'(wtd)':>8} {'':>6} {'':>7} {'@t*':>9} {'':>9}")
    print(f"{'='*92}")
    for name, mb in models_bar.items():
        print(f"  {name:<32} {mb['accuracy']:>8.4f} {mb['f1']:>8.4f} "
              f"{(mb['roc_auc'] or 0):>6.3f} {(mb['pr_auc'] or 0):>7.3f} "
              f"{(mb.get('tuned_f1_binary_pos') or 0):>9.4f} "
              f"{mb.get('train_time_sec', 0):>9.1f}")
    print(f"{'='*92}")
    print("  *F1 = weighted; AUC = ROC area; PR-AUC = average precision; "
          "TunedF1b = F1 on PNEUMONIA at val-tuned threshold.\n")

    plot_comparison_bar(models_bar,
        title=f"Approach 1 vs 2 — {cfg.DATASET_NAME} [{cfg.MODEL_NAME}] (weighted metrics)",
        save_path=os.path.join(cfg.RESULTS_DIR, "comparison_metrics.png"))

    # ── Training time ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]
    times = [m.get("train_time_sec", 0) for m in models_bar.values()]
    bars  = ax.bar(list(models_bar.keys()), times,
                   color=palette[:len(times)], alpha=0.85,
                   edgecolor="white", width=0.5)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{t:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Training Time (seconds)", fontsize=12)
    ax.set_title("Training Time Comparison", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_DIR, "comparison_time.png"), dpi=150)
    plt.show()
    print("[✓] Saved: comparison_time.png")

    # ── Confusion-matrix collage ────────────────────────────────────
    cm_files = [("SVM", "cm_svm.png"), ("LR", "cm_lr.png")]
    if "mlp" in ml:
        cm_files.append(("MLP", "cm_mlp.png"))
    cm_files.append(("End-to-End DL", "cm_dl.png"))

    n = len(cm_files)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Confusion Matrices — {cfg.MEMBER_NAME}", fontsize=14, fontweight="bold")
    for ax, (title, fname) in zip(axes, cm_files):
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

    # ── ROC / PR / Calibration / Threshold-sweep plots ──────────────
    if proba_pairs:
        plot_roc(proba_pairs,
                 title=f"ROC — {cfg.DATASET_NAME} [{cfg.MODEL_NAME}]",
                 save_path=os.path.join(cfg.RESULTS_DIR, "roc_curves.png"))
        plot_pr(proba_pairs,
                title=f"Precision-Recall — {cfg.DATASET_NAME} [{cfg.MODEL_NAME}]",
                save_path=os.path.join(cfg.RESULTS_DIR, "pr_curves.png"))

    dl_label = f"End-to-End {cfg.MODEL_NAME}"
    if dl_label in proba_pairs:
        y_test, p_test = proba_pairs[dl_label]
        t_star = enriched_models[dl_label].get("tuned_threshold", 0.5)
        plot_threshold_sweep(y_test, p_test, pos_label=1,
            title=f"Threshold Sweep (test) — {dl_label}",
            highlight=t_star,
            save_path=os.path.join(cfg.RESULTS_DIR, "threshold_sweep_dl.png"))
        plot_calibration(y_test, p_test,
            title=f"Calibration (test) — {dl_label}",
            save_path=os.path.join(cfg.RESULTS_DIR, "calibration_dl.png"))

        # Tuned-threshold confusion matrix for the DL model.
        y_pred_tuned = (p_test >= t_star).astype(int)
        plot_confusion_matrix(y_test, y_pred_tuned, cfg.CLASS_NAMES,
            title=f"DL Confusion Matrix @ thr={t_star:.2f} — {cfg.MEMBER_NAME}",
            save_path=os.path.join(cfg.RESULTS_DIR, "cm_dl_tuned.png"))

    # ── Comparative summary text ────────────────────────────────────
    best_acc  = max(models_bar, key=lambda k: models_bar[k]["accuracy"] or 0)
    best_auc  = max(models_bar, key=lambda k: models_bar[k].get("roc_auc") or 0)
    fastest   = min(models_bar, key=lambda k: models_bar[k].get("train_time_sec", 1e12))
    best_recall_pos = max(
        models_bar,
        key=lambda k: models_bar[k].get("tuned_recall_binary_pos",
                                         models_bar[k].get("recall_binary_pos") or 0)
    )

    summary = (
        f"COMPARATIVE ANALYSIS — {cfg.MEMBER_NAME}\n"
        f"Dataset: {cfg.DATASET_NAME}  |  Model: {cfg.MODEL_NAME}\n"
        f"Headline metrics: weighted Precision/Recall/F1. "
        f"Threshold tuned on val (F1 on PNEUMONIA).\n"
        f"{'='*72}\n"
    )
    for name, mb in models_bar.items():
        summary += (
            f"{name:<34} Acc:{mb['accuracy']:.4f}  F1_w:{mb['f1']:.4f}  "
            f"AUC:{(mb['roc_auc'] or 0):.4f}  PR-AUC:{(mb['pr_auc'] or 0):.4f}\n"
            f"{'':34} F1_macro:{(mb.get('f1_macro') or 0):.4f}  "
            f"F1_pos@0.5:{(mb.get('f1_binary_pos') or 0):.4f}  "
            f"thr*:{(mb.get('tuned_threshold') or 0.5):.2f}  "
            f"F1_pos@thr*:{(mb.get('tuned_f1_binary_pos') or 0):.4f}  "
            f"Time:{mb.get('train_time_sec', 0):.1f}s\n"
        )
    summary += (
        f"\nBest accuracy        : {best_acc}\n"
        f"Best ROC AUC         : {best_auc}\n"
        f"Best PNEUMONIA recall (tuned) : {best_recall_pos}\n"
        f"Fastest              : {fastest}\n"
    )

    summary += (
        f"\nWhat we'd ship:\n"
        f"  Use {best_auc} (highest AUC, threshold-agnostic ranking) at the "
        f"val-tuned threshold for clinical-style triage. The end-to-end model "
        f"benefits from the shared Phase-1/Phase-2 checkpoint fix, BatchNorm "
        f"frozen during fine-tuning, and label smoothing; ML baselines benefit "
        f"from L2-normalised GAP+GMP features and CV-tuned C/kernel.\n"
    )

    with open(os.path.join(cfg.RESULTS_DIR, "comparison_summary.txt"), "w") as f:
        f.write(summary)
    print(f"[✓] Saved: comparison_summary.txt\n[✓] Evaluation complete.\n")


if __name__ == "__main__":
    run()
