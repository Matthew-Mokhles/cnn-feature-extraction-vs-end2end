"""
Optional stratified split: move or copy a fraction of images from train/ to val/
per class under members/Matthew_xray/data/chest_xray/.

Example (10% from each class into val, copy — dry run first):
    python tools/split_train_val.py --fraction 0.10 --copy --dry-run
    python tools/split_train_val.py --fraction 0.10 --copy

Requires: train/<NORMAL|PNEUMONIA>/... and existing val/<class>/ folders.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil

IMAGE_EXT = (".jpg", ".jpeg", ".png")


def _images(folder: str) -> list[str]:
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXT)
    )


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    data_root = os.path.join(root, "data", "chest_xray")

    ap = argparse.ArgumentParser(description="Stratified train→val split helper")
    ap.add_argument("--fraction", type=float, default=0.15, help="Fraction per class to take from train (default 0.15)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true", help="Copy instead of move")
    ap.add_argument("--dry-run", action="store_true", help="Print actions only")
    args = ap.parse_args()

    if not (0 < args.fraction < 1):
        raise SystemExit("--fraction must be between 0 and 1 (exclusive).")

    train_root = os.path.join(data_root, "train")
    val_root = os.path.join(data_root, "val")
    if not os.path.isdir(train_root):
        raise SystemExit(f"Missing train folder: {train_root}")

    random.seed(args.seed)
    moved = 0
    for cls in os.listdir(train_root):
        td = os.path.join(train_root, cls)
        vd = os.path.join(val_root, cls)
        if not os.path.isdir(td):
            continue
        os.makedirs(vd, exist_ok=True)
        files = _images(td)
        k = max(1, int(round(len(files) * args.fraction)))
        k = min(k, len(files))
        pick = random.sample(files, k)
        verb = "Would copy" if args.dry_run and args.copy else ("Would move" if args.dry_run else ("Copy" if args.copy else "Move"))
        print(f"{verb} {k}/{len(files)} from train/{cls}/ → val/{cls}/")
        for fn in pick:
            src, dst = os.path.join(td, fn), os.path.join(vd, fn)
            if args.dry_run:
                print(f"  {src} → {dst}")
                continue
            if args.copy:
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)
            moved += 1

    if not args.dry_run:
        print(f"\nDone. {'Copied' if args.copy else 'Moved'} {moved} files total.")
    else:
        print("\nDry run — no files changed.")


if __name__ == "__main__":
    main()
