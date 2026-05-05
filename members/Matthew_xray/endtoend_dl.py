"""
members/Matthew_xray/endtoend_dl.py — Step 5, fully standalone.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config as cfg
from preprocessing import warn_if_validation_too_small
from utils.metrics import compute_metrics
from utils.visualization import plot_confusion_matrix, plot_training_curves


def build_model(freeze_all=True):
    base = cfg.MODEL_BUILDER(weights="imagenet", include_top=False,
                             pooling="avg", input_shape=cfg.INPUT_SHAPE)
    if freeze_all:
        base.trainable = False
    else:
        for layer in base.layers[:-20]: layer.trainable = False
        for layer in base.layers[-20:]: layer.trainable = True
    x = Dense(256, activation="relu")(base.output)
    x = Dropout(cfg.DROPOUT_RATE)(x)
    out = Dense(1, activation="sigmoid")(x)
    return Model(inputs=base.input, outputs=out)


def _callbacks():
    return [
        ModelCheckpoint(cfg.DL_MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
    ]


def _make_gens():
    train_dg = ImageDataGenerator(preprocessing_function=cfg.PREPROCESS_FN, **cfg.AUGMENTATION)
    eval_dg  = ImageDataGenerator(preprocessing_function=cfg.PREPROCESS_FN)
    common   = dict(target_size=cfg.IMG_SIZE, batch_size=cfg.BATCH_SIZE,
                    class_mode=cfg.CLASS_MODE, classes=cfg.CLASS_NAMES)
    return (train_dg.flow_from_directory(cfg.TRAIN_DIR, shuffle=True,  seed=42, **common),
            eval_dg.flow_from_directory(cfg.VAL_DIR,   shuffle=False, **common),
            eval_dg.flow_from_directory(cfg.TEST_DIR,  shuffle=False, **common))


def _train_label_array():
    """Integer labels 0..NUM_CLASSES-1 aligned with CLASS_NAMES / flow_from_directory order."""
    y = []
    for class_idx, cls in enumerate(cfg.CLASS_NAMES):
        d = os.path.join(cfg.TRAIN_DIR, cls)
        if not os.path.isdir(d):
            continue
        n = sum(1 for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        y.extend([class_idx] * n)
    return np.asarray(y, dtype=int)


def _class_weight_dict():
    y = _train_label_array()
    if y.size == 0:
        return None
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(wi) for c, wi in zip(classes, w)}


def run():
    train_gen, val_gen, test_gen = _make_gens()
    warn_if_validation_too_small(val_gen.samples)
    class_weight = _class_weight_dict()
    if class_weight:
        print(f"\n[*] Using balanced class_weight for training: {class_weight}")

    total_time = 0

    # Phase 1 — head only
    print(f"\n[{cfg.MEMBER_NAME}] Phase 1 — training head (base frozen)...")
    model = build_model(freeze_all=True)
    model.compile(optimizer=Adam(cfg.LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    t0 = time.time()
    h1 = model.fit(train_gen, validation_data=val_gen,
                   epochs=max(1, cfg.EPOCHS - 5), callbacks=_callbacks(),
                   class_weight=class_weight)
    total_time += time.time() - t0

    # Phase 2 — fine-tune last 20 layers
    print(f"\n[{cfg.MEMBER_NAME}] Phase 2 — fine-tuning last 20 layers...")
    for layer in model.layers[:-25]: layer.trainable = False
    for layer in model.layers[-25:]: layer.trainable = True
    model.compile(optimizer=Adam(cfg.LEARNING_RATE / 10), loss="binary_crossentropy", metrics=["accuracy"])
    t0 = time.time()
    h2 = model.fit(train_gen, validation_data=val_gen, epochs=5, callbacks=_callbacks(),
                     class_weight=class_weight)
    total_time += time.time() - t0
    print(f"  Total training time: {total_time:.0f}s")

    full_hist = {k: h1.history[k] + h2.history[k] for k in h1.history}
    full_hist["train_time_sec"] = total_time
    np.save(cfg.DL_HISTORY_PATH, full_hist)

    plot_training_curves(full_hist, title=f"End-to-End {cfg.MODEL_NAME} — {cfg.MEMBER_NAME}",
                         save_path=os.path.join(cfg.RESULTS_DIR, "dl_training_curves.png"))

    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = (y_prob.ravel() > 0.5).astype(int)
    y_true = test_gen.classes

    m = compute_metrics(y_true, y_pred, model_name=f"End-to-End {cfg.MODEL_NAME}")
    m["train_time_sec"] = total_time
    np.save(os.path.join(cfg.RESULTS_DIR, "dl_results.npy"), m)

    plot_confusion_matrix(y_true, y_pred, cfg.CLASS_NAMES,
        title=f"End-to-End Confusion Matrix — {cfg.MEMBER_NAME}",
        save_path=os.path.join(cfg.RESULTS_DIR, "cm_dl.png"))

    print(f"\n[✓] End-to-End DL done. Results → {cfg.RESULTS_DIR}\n")


if __name__ == "__main__":
    run()
