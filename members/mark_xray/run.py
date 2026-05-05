"""
members/mark_xray/run.py — Entry point for Mark.

    python main.py --member mark --step all
    python main.py --member mark --step 1

Steps:  1=extract_features  2=ml_classifier  3=endtoend_dl  4=evaluate
"""

import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import members.mark_xray.config as cfg
from core.feature_extractor import build_extractor, extract_and_save
from core.ml_trainer        import run_ml_pipeline
from core.dl_trainer        import run_dl_pipeline
from core.visualization     import plot_comparison_bar
import numpy as np


def step1():
    print(f"\n[{cfg.MEMBER_NAME}] Extracting features with {cfg.MODEL_NAME}...")
    extract_and_save(build_extractor(cfg), cfg)

def step2():
    run_ml_pipeline(cfg)

def step3():
    run_dl_pipeline(cfg)

def step4():
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
    print(f"\n[✓] Evaluation complete for {cfg.MEMBER_NAME}.\n")


STEPS = {"1": step1, "2": step2, "3": step3, "4": step4}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="all")
    args = parser.parse_args()
    if args.step == "all":
        for fn in STEPS.values(): fn()
    elif args.step in STEPS:
        STEPS[args.step]()
    else:
        print(f"Unknown step '{args.step}'. Options: {list(STEPS.keys())} or 'all'")
