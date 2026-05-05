"""utils/metrics.py — Mark's local copy."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred, model_name="Model"):
    avg = "binary" if len(set(y_true)) == 2 else "weighted"
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec  = recall_score(y_true, y_pred,    average=avg, zero_division=0)
    f1   = f1_score(y_true, y_pred,        average=avg, zero_division=0)
    print(f"\n  ── {model_name} ──")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1)
