"""
members/friend_b_cifar10/run.py  — Entry point for Friend B (CIFAR-10).
Usage:  python members/friend_b_cifar10/run.py [--step STEP]
"""
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import members.friend_b_cifar10.config as cfg


def step_extract_features():
    from core.feature_extractor import build_extractor, extract_and_save
    extract_and_save(build_extractor(cfg), cfg)

def step_ml_classifier():
    from core.ml_trainer import run_ml_pipeline
    # SVM on 10-class: use LR which handles multi-class natively
    run_ml_pipeline(cfg)

def step_dl_endtoend():
    from core.dl_trainer import run_dl_pipeline
    run_dl_pipeline(cfg)

def step_evaluate():
    import numpy as np
    from core.visualization import plot_comparison_bar
    ml = np.load(os.path.join(cfg.RESULTS_DIR, "ml_results.npy"), allow_pickle=True).item()
    dl = np.load(os.path.join(cfg.RESULTS_DIR, "dl_results.npy"), allow_pickle=True).item()
    models = {
        f"SVM ({cfg.MODEL_NAME} feats)": ml["svm"],
        f"LR  ({cfg.MODEL_NAME} feats)": ml["lr"],
        f"End-to-End {cfg.MODEL_NAME}":  dl,
    }
    plot_comparison_bar(models,
        title=f"Approach 1 vs 2 — {cfg.MEMBER_NAME} ({cfg.DATASET_NAME})",
        save_path=os.path.join(cfg.RESULTS_DIR, "comparison_metrics.png"))

STEPS = {"1": step_extract_features, "2": step_ml_classifier,
         "3": step_dl_endtoend,      "4": step_evaluate}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="all")
    args = parser.parse_args()
    if args.step == "all":
        for fn in STEPS.values(): fn()
    elif args.step in STEPS:
        STEPS[args.step]()
