"""
utils/metrics.py — Matthew's local copy, no shared dependencies.

Headline metrics use sklearn **weighted** precision/recall/F1 (better under
class imbalance). Binary metrics use **pos_label=1** (PNEUMONIA in
NORMAL/PNEUMONIA alphabetical folder order).

``compute_metrics_with_proba`` adds probability-aware diagnostics:
  - ROC AUC, PR AUC
  - F1-optimal threshold (sweep over [0.05, 0.95])
  - metrics re-evaluated at the tuned threshold
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)


def compute_metrics(y_true, y_pred, model_name="Model", pos_label=1):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    acc = accuracy_score(y_true, y_pred)

    prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    uniq = np.unique(np.concatenate([y_true, y_pred]))
    if len(uniq) >= 2 and pos_label in uniq:
        prec_bin = precision_score(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0)
        rec_bin = recall_score(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0)
        f1_bin = f1_score(y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0)
    else:
        prec_bin = rec_bin = f1_bin = 0.0

    print(f"\n  -- {model_name} --")
    print(f"  Accuracy           : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision (weighted): {prec_w:.4f}")
    print(f"  Recall (weighted)   : {rec_w:.4f}")
    print(f"  F1 (weighted)       : {f1_w:.4f}")
    print(f"  F1 (macro)          : {f1_macro:.4f}")
    print(f"  Prec/Rec/F1 binary (pos_label={pos_label}): {prec_bin:.4f} / {rec_bin:.4f} / {f1_bin:.4f}")

    return dict(
        accuracy=acc,
        precision=prec_w,
        recall=rec_w,
        f1=f1_w,
        precision_weighted=prec_w,
        recall_weighted=rec_w,
        f1_weighted=f1_w,
        precision_binary_pos=prec_bin,
        recall_binary_pos=rec_bin,
        f1_binary_pos=f1_bin,
        f1_macro=f1_macro,
    )


def find_best_threshold(y_true, y_proba, pos_label=1, criterion="f1",
                        grid=None):
    """Sweep decision thresholds and return the one maximising ``criterion``.

    criterion: 'f1' (binary, pos_label) or 'youden' (TPR-FPR).
    Returns (best_threshold, best_score).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = np.asarray(y_proba).astype(float).ravel()
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)

    best_t, best_s = 0.5, -np.inf
    for t in grid:
        y_pred = (y_proba >= t).astype(int)
        if criterion == "youden":
            tp = int(((y_pred == pos_label) & (y_true == pos_label)).sum())
            fn = int(((y_pred != pos_label) & (y_true == pos_label)).sum())
            fp = int(((y_pred == pos_label) & (y_true != pos_label)).sum())
            tn = int(((y_pred != pos_label) & (y_true != pos_label)).sum())
            tpr = tp / max(1, tp + fn)
            fpr = fp / max(1, fp + tn)
            s = tpr - fpr
        else:
            s = f1_score(y_true, y_pred, average="binary",
                         pos_label=pos_label, zero_division=0)
        if s > best_s:
            best_s, best_t = float(s), float(t)
    return best_t, best_s


def compute_metrics_with_proba(y_true, y_proba, model_name="Model",
                               pos_label=1, threshold=0.5,
                               tuned_threshold=None):
    """Probability-aware extension of compute_metrics.

    If ``tuned_threshold`` is provided (e.g. picked on val), the function also
    reports metrics at that threshold under ``tuned_*`` keys. Otherwise it
    finds the F1-optimal threshold on (y_true, y_proba) itself — useful for
    val-side tuning or quick diagnostics on test.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = np.asarray(y_proba).astype(float).ravel()
    out = {}

    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))
    except ValueError:
        out["pr_auc"] = float("nan")

    y_pred_default = (y_proba >= threshold).astype(int)
    base = compute_metrics(y_true, y_pred_default,
                           model_name=f"{model_name} @ thr={threshold:.2f}",
                           pos_label=pos_label)
    out.update(base)
    out["threshold"] = float(threshold)

    if tuned_threshold is None:
        tuned_threshold, _ = find_best_threshold(y_true, y_proba,
                                                 pos_label=pos_label,
                                                 criterion="f1")
    y_pred_tuned = (y_proba >= tuned_threshold).astype(int)
    tuned = compute_metrics(y_true, y_pred_tuned,
                            model_name=f"{model_name} @ thr={tuned_threshold:.2f} (tuned)",
                            pos_label=pos_label)
    out["tuned_threshold"] = float(tuned_threshold)
    for k, v in tuned.items():
        out[f"tuned_{k}"] = v

    print(f"  ROC AUC            : {out['roc_auc']:.4f}")
    print(f"  PR  AUC            : {out['pr_auc']:.4f}")
    return out
