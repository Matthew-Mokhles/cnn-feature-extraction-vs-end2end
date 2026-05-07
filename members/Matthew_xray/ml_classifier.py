"""
members/Matthew_xray/ml_classifier.py — Step 4, fully standalone.

Pipeline per model:
    StandardScaler → L2 row-normalisation → estimator (SVM | LR | MLP)

For SVM and LR we run StratifiedKFold GridSearchCV on the concatenated
(train+val) features and refit the best estimator on the full training set
before scoring on the held-out test split. The MLP head uses its own
internal early stopping (validation_fraction=0.1 carved out of train+val).

All best params and CV scores are written to ``results/ml_best_params.json``.
Probabilities for each model are persisted alongside labels so the Step-6
evaluation can compute ROC/PR/AUC and tune decision thresholds on val.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
import config as cfg
from utils.metrics import compute_metrics, compute_metrics_with_proba
from utils.visualization import plot_confusion_matrix


def _build_pipeline(estimator):
    """StandardScaler centres but does not unit-norm; a row-wise L2 step
    afterwards is critical for embedding-style features. ``Normalizer`` is
    used (rather than a lambda + FunctionTransformer) so the fitted
    pipeline is picklable by joblib."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("l2",    Normalizer(norm="l2")),
        ("clf",   estimator),
    ])


def _grid_search(name, base_estimator, param_grid, X, y, n_splits):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pipe = _build_pipeline(base_estimator)
    scoring = getattr(cfg, "ML_SCORING", "f1_macro")
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv,
                      scoring=scoring, n_jobs=-1, refit=True, verbose=1)
    print(f"\n[*] {name}: GridSearchCV ({n_splits}-fold, scoring={scoring}) over {param_grid}")
    t0 = time.time()
    gs.fit(X, y)
    elapsed = time.time() - t0
    print(f"    best params       : {gs.best_params_}")
    print(f"    best CV {scoring}: {gs.best_score_:.4f}   (fit {elapsed:.1f}s)")
    return gs, elapsed


def _proba(model, X):
    """Robust probability extraction (SVC requires probability=True; otherwise
    fall back to decision_function squashed through a logistic)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        d = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-d))
    return model.predict(X).astype(float)


def run():
    print(f"\n[{cfg.MEMBER_NAME}] Loading features...")
    X_train = np.load(cfg.FEATURES_TRAIN); y_train = np.load(cfg.LABELS_TRAIN).astype(int)
    X_val   = np.load(cfg.FEATURES_VAL);   y_val   = np.load(cfg.LABELS_VAL).astype(int)
    X_test  = np.load(cfg.FEATURES_TEST);  y_test  = np.load(cfg.LABELS_TEST).astype(int)

    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    print(f"    Train+Val features: {X_tv.shape}   Test features: {X_test.shape}")

    # CV must have at least one positive sample per class per fold.
    n_splits = max(2, min(cfg.ML_KFOLDS, int(np.bincount(y_tv).min())))
    if n_splits != cfg.ML_KFOLDS:
        print(f"[!] Reducing CV folds from {cfg.ML_KFOLDS} to {n_splits} "
              f"(class imbalance constraint).")

    grids = {
        "svm": (
            SVC(probability=True, random_state=42),
            {
                "clf__C":            cfg.ML_GRID_SVM_C,
                "clf__kernel":       cfg.ML_GRID_SVM_KERNELS,
                "clf__class_weight": cfg.ML_GRID_SVM_CLASS_WEIGHT,
            },
        ),
        "lr": (
            LogisticRegression(max_iter=cfg.LR_MAX_ITER, random_state=42),
            {
                "clf__C":            cfg.ML_GRID_LR_C,
                "clf__class_weight": cfg.ML_GRID_LR_CLASS_WEIGHT,
            },
        ),
    }

    results, best_params = {}, {}

    for name, (est, grid) in grids.items():
        gs, elapsed = _grid_search(name.upper(), est, grid, X_tv, y_tv, n_splits)
        best = gs.best_estimator_

        preds  = best.predict(X_test)
        proba  = _proba(best, X_test)
        proba_val = _proba(best, X_val)

        m = compute_metrics(y_test, preds, model_name=f"{name.upper()} ({cfg.MEMBER_NAME})")
        m_p = compute_metrics_with_proba(y_test, proba,
                                         model_name=f"{name.upper()} (proba)")
        m.update({k: v for k, v in m_p.items() if k not in m})
        m["train_time_sec"]  = elapsed
        m["best_params"]     = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
        m["cv_best_f1_w"]    = float(gs.best_score_)
        results[name]        = m
        best_params[name]    = m["best_params"] | {"cv_best_f1_w": m["cv_best_f1_w"]}

        joblib.dump(best, os.path.join(cfg.MODELS_DIR, f"{name}_classifier.joblib"))
        np.save(os.path.join(cfg.RESULTS_DIR, f"{name}_test_proba.npy"),
                np.column_stack([y_test, proba]))
        np.save(os.path.join(cfg.RESULTS_DIR, f"{name}_val_proba.npy"),
                np.column_stack([y_val, proba_val]))
        print(classification_report(y_test, preds, target_names=cfg.CLASS_NAMES))
        plot_confusion_matrix(y_test, preds, cfg.CLASS_NAMES,
            title=f"{name.upper()} Confusion Matrix — {cfg.MEMBER_NAME}",
            save_path=os.path.join(cfg.RESULTS_DIR, f"cm_{name}.png"))

    # ── MLP head: fit once with internal validation ─────────────────────
    print("\n[*] MLP: fitting MLPClassifier(256, 64) with early stopping...")
    mlp_pipe = _build_pipeline(MLPClassifier(
        hidden_layer_sizes=(256, 64), activation="relu", solver="adam",
        alpha=1e-4, batch_size=128, max_iter=200,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=10, random_state=42,
    ))
    t0 = time.time()
    mlp_pipe.fit(X_tv, y_tv)
    elapsed = time.time() - t0
    preds = mlp_pipe.predict(X_test)
    proba = _proba(mlp_pipe, X_test)
    proba_val = _proba(mlp_pipe, X_val)
    m = compute_metrics(y_test, preds, model_name=f"MLP ({cfg.MEMBER_NAME})")
    m_p = compute_metrics_with_proba(y_test, proba, model_name="MLP (proba)")
    m.update({k: v for k, v in m_p.items() if k not in m})
    m["train_time_sec"] = elapsed
    results["mlp"]      = m
    joblib.dump(mlp_pipe, cfg.MLP_MODEL_PATH)
    np.save(os.path.join(cfg.RESULTS_DIR, "mlp_test_proba.npy"),
            np.column_stack([y_test, proba]))
    np.save(os.path.join(cfg.RESULTS_DIR, "mlp_val_proba.npy"),
            np.column_stack([y_val, proba_val]))
    print(classification_report(y_test, preds, target_names=cfg.CLASS_NAMES))
    plot_confusion_matrix(y_test, preds, cfg.CLASS_NAMES,
        title=f"MLP Confusion Matrix — {cfg.MEMBER_NAME}",
        save_path=os.path.join(cfg.RESULTS_DIR, "cm_mlp.png"))

    np.save(os.path.join(cfg.RESULTS_DIR, "ml_results.npy"), results)
    with open(os.path.join(cfg.RESULTS_DIR, "ml_best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    print(f"\n[✓] ML pipeline done. Results → {cfg.RESULTS_DIR}\n")


if __name__ == "__main__":
    run()
