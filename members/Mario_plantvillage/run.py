"""
run.py — Mario's standalone entry point.
Run from this folder:  python run.py --step all
Or from repo root:     python members/Mario_plantvillage/run.py --step all

Steps: 1=EDA  2=preprocessing  3=features  4=ML  5=DL  6=evaluation
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_analysis   import run as step1
from preprocessing      import run as step2
from feature_extraction import run as step3
from ml_classifier      import run as step4
from endtoend_dl        import run as step5
from evaluation         import run as step6

STEPS = {"1": step1, "2": step2, "3": step3, "4": step4, "5": step5, "6": step6}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mario — PlantVillage + ResNet50")
    parser.add_argument("--step", default="all", help="1-6 or 'all'")
    args = parser.parse_args()
    if args.step == "all":
        for fn in STEPS.values(): fn()
    elif args.step in STEPS:
        STEPS[args.step]()
    else:
        print(f"Unknown step. Choose from {list(STEPS.keys())} or 'all'")
