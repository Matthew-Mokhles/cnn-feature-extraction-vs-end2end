"""
main.py — Root dispatcher for all team members.
────────────────────────────────────────────────────────────────────────────────
Usage:
    python main.py --member matthew --step all
    python main.py --member matthew --step 1

Members status:
    matthew  →  members/Matthew_xray/        Chest X-Ray   + ResNet50  (READY)
    mark     →  members/mark_xray/           Chest X-Ray   + mobilenet    (WIP — owner: Mark)
    mario    →  members/Mario_plantvillage/  PlantVillage  + EfficientNetB0   (WIP — owner: Mario)
    veronia  →  members/Veronia_cifar10/     CIFAR-10      + EfficientNetB0  (WIP — owner: Veronia)

Only members listed in MEMBER_MAP below are runnable from this dispatcher.
The WIP entries above are intentionally omitted until each owner finishes
their standalone pipeline (config.py + step files + utils/).

Steps (Matthew):
    1  dataset_analysis
    2  preprocessing visualisation
    3  feature_extraction
    4  ml_classifier
    5  endtoend_dl
    6  evaluation
    all → 1-6
"""

import os
import sys
import argparse
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MEMBER_MAP = {
    "matthew": "members.Matthew_xray.run",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Team Project Runner")
    parser.add_argument("--member", required=True, choices=MEMBER_MAP.keys(),
                        help=f"Team member: {list(MEMBER_MAP.keys())}")
    parser.add_argument("--step", default="all",
                        help="Step number or 'all' (default: all)")
    args = parser.parse_args()

    mod = importlib.import_module(MEMBER_MAP[args.member])

    if args.step == "all":
        for fn in mod.STEPS.values():
            fn()
    elif args.step in mod.STEPS:
        mod.STEPS[args.step]()
    else:
        print(f"[!] Unknown step '{args.step}' for member '{args.member}'.")
        print(f"    Available: {list(mod.STEPS.keys())} or 'all'")
        sys.exit(1)
