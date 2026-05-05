"""
members/Matthew_xray/ml_classifier.py — Step 4, fully standalone.

Feature vectors from train/ and val/ are **concatenated** and used to fit SVM
and logistic regression. There is **no separate validation holdout** for ML
hyperparameter search in this baseline — reported metrics are computed only on
the **held-out test** features (same split as image folders).

For stronger methodology, use StratifiedKFold on the concatenated training
features to tune `SVM_C` / `LR_C`, then refit on all training features before a
single final test evaluation.

Run via: python main.py --member matthew --step 4
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time, numpy as np, joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import config as cfg
from utils.metrics import compute_metrics
from utils.visualization import plot_confusion_matrix


def run():
    print(f"\n[{cfg.MEMBER_NAME}] Loading features...")
    X_train = np.load(cfg.FEATURES_TRAIN); y_train = np.load(cfg.LABELS_TRAIN).astype(int)
    X_val   = np.load(cfg.FEATURES_VAL);   y_val   = np.load(cfg.LABELS_VAL).astype(int)
    X_test  = np.load(cfg.FEATURES_TEST);  y_test  = np.load(cfg.LABELS_TEST).astype(int)

    X_tv = np.vstack([X_train, X_val]); y_tv = np.concatenate([y_train, y_val])

    scaler = StandardScaler()
    X_tv_sc   = scaler.fit_transform(X_tv)
    X_test_sc = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(cfg.MODELS_DIR, "scaler.joblib"))

    results = {}
    for name, clf in [
        ("svm", SVC(kernel=cfg.SVM_KERNEL, C=cfg.SVM_C, random_state=42,
                    probability=True, class_weight="balanced")),
        ("lr",  LogisticRegression(C=cfg.LR_C, max_iter=cfg.LR_MAX_ITER, random_state=42,
                                   class_weight="balanced")),
    ]:
        print(f"\n[*] Training {name.upper()}...")
        t0 = time.time()
        clf.fit(X_tv_sc, y_tv)
        elapsed = time.time() - t0
        preds = clf.predict(X_test_sc)
        m = compute_metrics(y_test, preds, model_name=f"{name.upper()} ({cfg.MEMBER_NAME})")
        m["train_time_sec"] = elapsed
        results[name] = m
        joblib.dump(clf, os.path.join(cfg.MODELS_DIR, f"{name}_classifier.joblib"))
        print(classification_report(y_test, preds, target_names=cfg.CLASS_NAMES))
        plot_confusion_matrix(y_test, preds, cfg.CLASS_NAMES,
            title=f"{name.upper()} Confusion Matrix — {cfg.MEMBER_NAME}",
            save_path=os.path.join(cfg.RESULTS_DIR, f"cm_{name}.png"))

    np.save(os.path.join(cfg.RESULTS_DIR, "ml_results.npy"), results)
    print(f"\n[✓] ML pipeline done. Results → {cfg.RESULTS_DIR}\n")


if __name__ == "__main__":
    run()
