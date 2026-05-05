"""
members/Matthew_xray/feature_extraction.py — Step 3, fully standalone.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config as cfg


def build_extractor():
    base = cfg.MODEL_BUILDER(weights="imagenet", include_top=False,
                             pooling="avg", input_shape=cfg.INPUT_SHAPE)
    base.trainable = False
    print(f"[✓] {cfg.MODEL_NAME} loaded — {len(base.layers)} layers frozen")
    print(f"    Output shape: {base.output_shape}")
    return base


def _extract_split(model, generator, desc):
    generator.reset()
    n_steps = int(np.ceil(generator.samples / generator.batch_size))
    feats, labels = [], []
    for _ in tqdm(range(n_steps), desc=desc, unit="batch"):
        xb, yb = next(generator)
        feats.append(model.predict(xb, verbose=0))
        labels.append(yb)
    return np.vstack(feats)[:generator.samples], np.concatenate(labels)[:generator.samples]


def run():
    print(f"\n[{cfg.MEMBER_NAME}] Extracting features with {cfg.MODEL_NAME}...")
    eval_datagen = ImageDataGenerator(preprocessing_function=cfg.PREPROCESS_FN)
    common = dict(target_size=cfg.IMG_SIZE, batch_size=cfg.BATCH_SIZE,
                  class_mode=cfg.CLASS_MODE, classes=cfg.CLASS_NAMES, shuffle=False)
    gens = {
        "Train": (eval_datagen.flow_from_directory(cfg.TRAIN_DIR, **common),
                  cfg.FEATURES_TRAIN, cfg.LABELS_TRAIN),
        "Val":   (eval_datagen.flow_from_directory(cfg.VAL_DIR,   **common),
                  cfg.FEATURES_VAL,   cfg.LABELS_VAL),
        "Test":  (eval_datagen.flow_from_directory(cfg.TEST_DIR,  **common),
                  cfg.FEATURES_TEST,  cfg.LABELS_TEST),
    }
    extractor = build_extractor()
    for tag, (gen, fp, lp) in gens.items():
        f, l = _extract_split(extractor, gen, desc=f"  {tag}")
        np.save(fp, f); np.save(lp, l)
        print(f"    {tag}: features={f.shape}")
    print(f"\n[✓] Features saved to: {cfg.FEATURES_DIR}\n")


if __name__ == "__main__":
    run()
