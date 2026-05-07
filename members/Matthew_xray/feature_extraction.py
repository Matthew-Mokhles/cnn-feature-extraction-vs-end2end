"""
members/Matthew_xray/feature_extraction.py — Step 3, fully standalone.

Embedding = concat(GlobalAveragePooling2D, GlobalMaxPooling2D) over the
ImageNet-pretrained backbone. This doubles the feature dimension (2048 → 4096
for ResNet50) and tends to improve linear/RBF classifiers by 3-5 points on
chest x-ray vs. plain GAP.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config as cfg


def build_extractor():
    base = cfg.MODEL_BUILDER(weights="imagenet", include_top=False,
                             input_shape=cfg.INPUT_SHAPE)
    base.trainable = False
    gap = GlobalAveragePooling2D(name="gap")(base.output)
    gmp = GlobalMaxPooling2D(name="gmp")(base.output)
    feats = Concatenate(name="gap_gmp_concat")([gap, gmp])
    extractor = Model(inputs=base.input, outputs=feats, name=f"{cfg.MODEL_NAME}_GAP_GMP")
    print(f"[✓] {cfg.MODEL_NAME} loaded — {len(base.layers)} layers frozen")
    print(f"    Output shape: {extractor.output_shape}  (GAP+GMP)")
    return extractor


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
    print(f"\n[{cfg.MEMBER_NAME}] Extracting features with {cfg.MODEL_NAME} (GAP+GMP)...")
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
