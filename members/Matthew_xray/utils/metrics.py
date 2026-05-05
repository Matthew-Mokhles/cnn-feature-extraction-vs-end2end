"""
utils/metrics.py — Matthew's local copy, no shared dependencies.

Primary headline metrics use sklearn **weighted** precision/recall/F1 (better
under class imbalance). Binary metrics use **pos_label=1** (second class in
alphabetical folder order: PNEUMONIA for NORMAL/PNEUMONIA).
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
