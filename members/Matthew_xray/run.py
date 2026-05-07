"""
members/Matthew_xray/run.py — Matthew's standalone entry point.
Run from inside this folder:   cd members/Matthew_xray && python run.py --step all
Or from repo root:             python members/Matthew_xray/run.py --step all

Steps: 1=EDA  2=preprocessing  3=features  4=ML  5=DL  6=evaluation
       7=evaluation (alias — re-runs Step 6 / threshold-tuned eval without
         retraining; useful after only updating ml_classifier.py or
         evaluation.py).
"""
import os
import sys

# Quieter TensorFlow C++ logs unless the user already set a level.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_analysis  import run as step1
from preprocessing     import run as step2
from feature_extraction import run as step3
from ml_classifier     import run as step4
from endtoend_dl       import run as step5
from evaluation        import run as step6

STEPS = {
    "1": step1, "2": step2, "3": step3,
    "4": step4, "5": step5, "6": step6,
    "7": step6,  # alias for re-running evaluation without retraining
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matthew — Chest X-Ray + ResNet50")
    parser.add_argument("--step", default="all",
                        help="1-6 (or 7=alias for 6) or 'all'")
    args = parser.parse_args()
    if args.step == "all":
        for key in ("1", "2", "3", "4", "5", "6"):
            STEPS[key]()
    elif args.step in STEPS:
        STEPS[args.step]()
    else:
        print(f"Unknown step. Choose from {list(STEPS.keys())} or 'all'")
