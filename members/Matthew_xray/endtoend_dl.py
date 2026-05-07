"""
members/Matthew_xray/endtoend_dl.py
════════════════════════════════════════════════════════════════════════════════
End-to-End ResNet50 fine-tuning — Step 5.

Two-phase training strategy
────────────────────────────
Phase 1  Head-only warm-up (base fully frozen).
         Runs for cfg.EPOCHS - cfg.PHASE2_EPOCHS epochs with Adam + constant
         LR + ReduceLROnPlateau.  Avoids corrupting ImageNet features before
         the new head has converged.

Phase 2  Selective base fine-tuning (last N base layers unfrozen).
         Uses cosine-decay LR delivered via LearningRateScheduler callback
         (NOT a CosineDecay schedule object inside the optimizer).

         ⚠  On TensorFlow 2.10.x / Windows the Keras Progbar tries to log
            the learning rate by calling  lr_schedule * batch_index  (int
            multiplication on the schedule object), which raises:
                TypeError: unsupported operand type(s) for *: 'CosineDecay' and 'int'
            Delivering the LR as a callback that returns a plain Python float
            each epoch is the correct workaround for TF ≤ 2.10.

         BatchNorm layers are kept in inference mode during Phase 2 to prevent
         the small fine-tuning batch from corrupting running statistics.

Robustness features
────────────────────
• Balanced class_weight — corrects the PNEUMONIA:NORMAL ≈ 3:1 imbalance.
• Label smoothing (cfg.LABEL_SMOOTHING) — prevents overconfident predictions.
• Phase-2 checkpoint seeded with Phase-1's best val_accuracy so the globally
  best model is always saved, regardless of which phase produced it.
• TTA at inference: averages probabilities over 4 deterministic views
  (original, H-flip, V-flip, 90° rotation) for a measurable accuracy boost
  with zero training cost.
• MixUp-aware: if cfg.MIXUP_ALPHA > 0 and tf.data is active, class_weight is
  disabled automatically (incompatible with soft labels).
• RESUME_FROM_PHASE1: skip Phase 1 entirely and load the saved checkpoint.
• Full history union merge: Phase-2-only keys ('lr') are never dropped.
════════════════════════════════════════════════════════════════════════════════
"""
import math
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization, Concatenate, Dense, Dropout,
    GlobalAveragePooling2D, GlobalMaxPooling2D,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config as cfg
from preprocessing import warn_if_validation_too_small
from utils.metrics import compute_metrics, compute_metrics_with_proba
from utils.visualization import plot_confusion_matrix, plot_training_curves


# ══════════════════════════════════════════════════════════════════════════════
# LR SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════

def _cosine_decay_callback(initial_lr: float, total_epochs: int,
                            alpha: float = 0.05) -> LearningRateScheduler:
    """
    Cosine-decay delivered as a *per-epoch callback* that returns a plain float.

    This is the only safe approach on TF 2.10.x / Windows.  Passing a
    CosineDecay *schedule object* directly into Adam() causes the Keras Progbar
    to compute  schedule * batch_index  (int), which raises:
        TypeError: unsupported operand type(s) for *: 'CosineDecay' and 'int'

    The callback path avoids this because Keras reads `optimizer.lr` (a
    tf.Variable holding a plain float) rather than the schedule object.
    """
    def _schedule(epoch: int, _current_lr: float) -> float:
        # epoch is 0-indexed inside the callback.
        progress = epoch / max(total_epochs - 1, 1)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(initial_lr * (alpha + (1.0 - alpha) * cosine))
    return LearningRateScheduler(_schedule, verbose=1)


# ══════════════════════════════════════════════════════════════════════════════
# SWA (Stochastic Weight Averaging)
# ══════════════════════════════════════════════════════════════════════════════

class _SWACallback(tf.keras.callbacks.Callback):
    """Stochastic Weight Averaging — averages weights from epoch ``start_epoch``
    onward, then writes the averaged weights back into the model at training end.

    This finds a flatter minimum that generalises better, especially when the
    test distribution differs from train/val (as in this chest-xray dataset).
    A pre-SWA snapshot is saved so both versions can be compared.
    """

    def __init__(self, start_epoch: int, save_pre_swa: str = ""):
        super().__init__()
        self.start_epoch  = start_epoch
        self.save_pre_swa = save_pre_swa
        self._weight_sum  = None
        self._n           = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch < self.start_epoch:
            return
        weights = self.model.get_weights()
        if self._weight_sum is None:
            self._weight_sum = [np.zeros_like(w) for w in weights]
        for i, w in enumerate(weights):
            self._weight_sum[i] += w
        self._n += 1

    def on_train_end(self, logs=None):
        if self._n == 0:
            print("[SWA] No epochs averaged — skipping.")
            return
        print(f"[SWA] Averaging weights from {self._n} epochs "
              f"(starting at epoch {self.start_epoch}).")
        if self.save_pre_swa:
            self.model.save(self.save_pre_swa)
            print(f"[SWA] Pre-SWA snapshot → {self.save_pre_swa}")
        avg_weights = [s / self._n for s in self._weight_sum]
        self.model.set_weights(avg_weights)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

# Head layer names are explicit so the Phase-2 unfreeze filter (HEAD_NAMES)
# can reliably exclude them.  Without explicit names Keras auto-assigns
# 'dense' / 'dense_1' — none of which match any named filter string.
HEAD_NAMES = {
    "head_gap", "head_gmp", "head_concat",
    "head_fc512", "head_bn1", "head_drop1",
    "head_fc128", "head_bn2", "head_drop2",
    "head_output",
}


def build_model() -> Model:
    """ResNet50 backbone (fully frozen) + GAP⊕GMP + deep regularised head.

    Architecture
    ────────────
    ResNet50 base (no top, no pooling)
      → GlobalAveragePooling2D  ─┐
      → GlobalMaxPooling2D      ─┤→ Concatenate (4096-d)
      → Dense(512) + BN + ReLU + Dropout(0.4)  + L2
      → Dense(128) + BN + ReLU + Dropout(0.3)  + L2
      → Dense(1, sigmoid)
    """
    base = cfg.MODEL_BUILDER(
        weights     = "imagenet",
        include_top = False,
        pooling     = None,          # no pooling — we add our own
        input_shape = cfg.INPUT_SHAPE,
    )
    base.trainable = False

    gap = GlobalAveragePooling2D(name="head_gap")(base.output)
    gmp = GlobalMaxPooling2D(name="head_gmp")(base.output)
    x   = Concatenate(name="head_concat")([gap, gmp])      # 4096-d

    x   = Dense(512, kernel_regularizer=l2(1e-4), name="head_fc512")(x)
    x   = BatchNormalization(name="head_bn1")(x)
    x   = tf.keras.activations.relu(x)
    x   = Dropout(cfg.DROPOUT_RATE, name="head_drop1")(x)   # 0.4

    x   = Dense(128, kernel_regularizer=l2(1e-4), name="head_fc128")(x)
    x   = BatchNormalization(name="head_bn2")(x)
    x   = tf.keras.activations.relu(x)
    x   = Dropout(max(cfg.DROPOUT_RATE - 0.1, 0.1), name="head_drop2")(x)

    out = Dense(1, activation="sigmoid", name="head_output")(x)
    return Model(inputs=base.input, outputs=out, name="resnet50_endtoend")


def _unfreeze_for_phase2(model: Model) -> None:
    """
    Unfreeze the last cfg.PHASE2_UNFREEZE layers of the ResNet50 BASE only.

    The naive approach  model.layers[-N:]  counts from the tail of the *full*
    model, which includes the 3 custom head layers (Dense/Dropout/Dense).
    With N=30 that means only 27 ResNet50 layers actually unfreeze instead of 30.

    Fix: build a list of base layers by excluding HEAD_NAMES, then slice that.
    """
    base_layers = [l for l in model.layers if l.name not in HEAD_NAMES]
    n           = min(cfg.PHASE2_UNFREEZE, len(base_layers))

    # Step 1: freeze everything.
    for layer in model.layers:
        layer.trainable = False

    # Step 2: unfreeze exactly N base layers.
    for layer in base_layers[-n:]:
        layer.trainable = True

    # Step 3: always keep the head trainable.
    for layer in model.layers:
        if layer.name in HEAD_NAMES:
            layer.trainable = True

    # Step 4: optionally freeze all BN layers (default True for fine-tuning).
    freeze_bn = getattr(cfg, "PHASE2_FREEZE_BN", True)
    if freeze_bn:
        for layer in model.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = False

    trainable   = sum(1 for l in model.layers if l.trainable)
    total       = len(model.layers)
    bn_frozen   = sum(1 for l in model.layers
                      if isinstance(l, BatchNormalization) and not l.trainable)
    print(f"  Trainable layers : {trainable}/{total}  "
          f"(BN frozen: {bn_frozen}  base unfrozen: {n})")


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def _make_checkpoint(initial_best: float = np.inf,
                     monitor: str = "val_loss") -> ModelCheckpoint:
    """
    ModelCheckpoint seeded with initial_best so Phase 2 cannot overwrite a
    better Phase-1 model with a worse one.  Monitors val_loss by default
    (smoother than val_accuracy on small validation sets).

    initial_value_threshold was added in TF 2.5; we emulate it on older builds.
    """
    mode = "min" if "loss" in monitor else "max"
    try:
        return ModelCheckpoint(
            cfg.DL_MODEL_PATH,
            monitor                 = monitor,
            mode                    = mode,
            save_best_only          = True,
            initial_value_threshold = initial_best,
            verbose                 = 1,
        )
    except TypeError:
        ckpt = ModelCheckpoint(cfg.DL_MODEL_PATH,
                               monitor=monitor, mode=mode,
                               save_best_only=True, verbose=1)
        if np.isfinite(initial_best):
            ckpt.best = initial_best
        return ckpt


# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════

def _make_gens():
    """ImageDataGenerator fallback when tf.data pipeline is unavailable."""
    train_dg = ImageDataGenerator(
        preprocessing_function = cfg.PREPROCESS_FN,
        **cfg.AUGMENTATION,
    )
    eval_dg = ImageDataGenerator(preprocessing_function=cfg.PREPROCESS_FN)
    common  = dict(
        target_size = cfg.IMG_SIZE,
        batch_size  = cfg.BATCH_SIZE,
        class_mode  = cfg.CLASS_MODE,
        classes     = cfg.CLASS_NAMES,
    )
    return (
        train_dg.flow_from_directory(cfg.TRAIN_DIR, shuffle=True,  seed=42, **common),
        eval_dg.flow_from_directory( cfg.VAL_DIR,   shuffle=False, **common),
        eval_dg.flow_from_directory( cfg.TEST_DIR,  shuffle=False, **common),
    )


def _train_label_array() -> np.ndarray:
    y = []
    for idx, cls in enumerate(cfg.CLASS_NAMES):
        d = os.path.join(cfg.TRAIN_DIR, cls)
        if not os.path.isdir(d):
            continue
        n = sum(1 for f in os.listdir(d)
                if f.lower().endswith((".jpg", ".jpeg", ".png")))
        y.extend([idx] * n)
    return np.asarray(y, dtype=int)


def _class_weight_dict():
    y = _train_label_array()
    if y.size == 0:
        return None
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    d = {int(c): float(wi) for c, wi in zip(classes, w)}
    return d


def _val_samples(val_obj) -> int:
    """
    Number of validation samples.
    Guards against tf.data.Dataset.cardinality() returning -1 (INFINITE)
    or -2 (UNKNOWN) — both give nonsensical negative sample counts if
    multiplied by BATCH_SIZE without checking.
    """
    if hasattr(val_obj, "samples"):
        return int(val_obj.samples)
    if hasattr(val_obj, "cardinality"):
        try:
            c = int(val_obj.cardinality().numpy())
            if c > 0:
                return c * cfg.BATCH_SIZE
        except Exception:
            pass
    return cfg.MIN_RECOMMENDED_VAL_SAMPLES


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE  (standard + TTA)
# ══════════════════════════════════════════════════════════════════════════════

def _tta_proba(model: Model, xb: np.ndarray) -> np.ndarray:
    """
    8-view Test-Time Augmentation (deterministic):
        1. original
        2. horizontal flip
        3. vertical flip
        4. 90° rotation (CCW)
        5–8. four corner crops at 0.9× scale, resized back

    Averages the sigmoid outputs across all views for more robust
    probability estimates.  Zero training cost.
    """
    h, w = xb.shape[1], xb.shape[2]
    ch, cw = int(h * 0.9), int(w * 0.9)

    def _crop_resize(arr, top, left):
        """Crop and resize back to original size via nearest-neighbour."""
        cropped = arr[:, top:top+ch, left:left+cw, :]
        # Use tf for resize (handles batched 4-d arrays cleanly).
        return tf.image.resize(cropped, [h, w], method="bilinear").numpy()

    views = [
        xb,                                           # 1. original
        xb[:, :, ::-1, :],                            # 2. H-flip
        xb[:, ::-1, :, :],                            # 3. V-flip
        np.rot90(xb, k=1, axes=(1, 2)),               # 4. 90° CCW
        _crop_resize(xb, 0,      0),                  # 5. top-left crop
        _crop_resize(xb, 0,      w - cw),             # 6. top-right crop
        _crop_resize(xb, h - ch, 0),                  # 7. bottom-left crop
        _crop_resize(xb, h - ch, w - cw),             # 8. bottom-right crop
    ]
    probs = [model.predict(v, verbose=0).ravel() for v in views]
    return np.mean(probs, axis=0)


def _predict_all(model: Model, data_obj, use_tta: bool = False):
    """
    Returns (y_true, y_proba) arrays for the complete dataset.
    Handles both Keras DirectoryIterator and tf.data.Dataset objects.
    """
    use_tta = use_tta and getattr(cfg, "USE_TTA", False)

    # ── Keras DirectoryIterator (fast path without TTA) ──────────────────
    if hasattr(data_obj, "classes") and not use_tta:
        data_obj.reset()
        y_proba = model.predict(data_obj, verbose=1).ravel()
        return data_obj.classes.astype(int), y_proba

    # ── Batch-by-batch path (TTA or tf.data) ─────────────────────────────
    if hasattr(data_obj, "classes"):
        data_obj.reset()
        n_steps = math.ceil(data_obj.samples / data_obj.batch_size)
        iterator = (next(data_obj) for _ in range(n_steps))
    else:
        iterator = iter(data_obj)

    y_true_parts, y_proba_parts = [], []
    for xb, yb in iterator:
        xb_np = xb if isinstance(xb, np.ndarray) else xb.numpy()
        yb_np = yb if isinstance(yb, np.ndarray) else yb.numpy()
        proba = _tta_proba(model, xb_np) if use_tta \
                else model.predict(xb_np, verbose=0).ravel()
        y_proba_parts.append(proba)
        y_true_parts.append(yb_np.astype(int).ravel())

    y_proba = np.concatenate(y_proba_parts)
    y_true  = np.concatenate(y_true_parts)

    # Trim padding introduced by batching.
    if hasattr(data_obj, "samples"):
        y_proba = y_proba[:data_obj.samples]
        y_true  = y_true[:data_obj.samples]

    return y_true, y_proba


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def _phase_summary(name: str, hist: dict, elapsed: float) -> float:
    """Print per-phase stats and return the best val_accuracy."""
    accs   = hist.get("val_accuracy", [])
    losses = hist.get("val_loss", [])
    lrs    = hist.get("lr", [])
    best_a = max(accs)   if accs   else float("nan")
    best_l = min(losses) if losses else float("nan")
    last_r = float(lrs[-1]) if lrs else float("nan")

    bar = "─" * 50
    print(f"\n  {bar}")
    print(f"  {name}")
    print(f"  {bar}")
    print(f"  Epochs trained    : {len(accs)}")
    print(f"  Best val_accuracy : {best_a:.4f}")
    print(f"  Best val_loss     : {best_l:.4f}")
    if not math.isnan(last_r):
        print(f"  Final LR          : {last_r:.3e}")
    print(f"  Wall time         : {elapsed:.0f}s")
    print(f"  {bar}\n")
    return best_a if accs else -np.inf


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run() -> None:
    # ── Reproducibility ────────────────────────────────────────────────────
    seed = getattr(cfg, "RANDOM_SEED", 42)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        tf.random.set_seed(seed)
    np.random.seed(seed)

    # ── Data ───────────────────────────────────────────────────────────────
    use_tf_data = getattr(cfg, "USE_TF_DATA", False)
    if use_tf_data:
        try:
            from data_pipeline import get_datasets
            train_gen, val_gen, test_gen = get_datasets()
            print("[*] Using tf.data pipeline (CLAHE + cache + prefetch).")
        except Exception as exc:
            print(f"[!] tf.data unavailable ({exc}); falling back to "
                  "ImageDataGenerator.")
            train_gen, val_gen, test_gen = _make_gens()
    else:
        train_gen, val_gen, test_gen = _make_gens()

    warn_if_validation_too_small(_val_samples(val_gen))

    # ── Class weight ───────────────────────────────────────────────────────
    class_weight = _class_weight_dict()
    using_tf_data    = isinstance(train_gen, tf.data.Dataset)
    mixup_alpha      = float(getattr(cfg, "MIXUP_ALPHA", 0.0) or 0.0)
    mixup_active     = mixup_alpha > 0.0 and using_tf_data

    if mixup_active:
        print("[*] MixUp active — class_weight disabled (incompatible with "
              "soft MixUp labels).")
        class_weight = None
    elif class_weight:
        print(f"[*] Balanced class_weight: {class_weight}")

    bce        = BinaryCrossentropy(
                     label_smoothing=getattr(cfg, "LABEL_SMOOTHING", 0.0))
    total_time = 0.0

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 1 — warm up the head (base fully frozen)
    # ──────────────────────────────────────────────────────────────────────
    resume         = getattr(cfg, "RESUME_FROM_PHASE1", False)
    phase1_epochs  = cfg.EPOCHS - cfg.PHASE2_EPOCHS

    if resume and os.path.exists(cfg.DL_MODEL_PHASE1_PATH):
        print(f"\n[*] RESUME_FROM_PHASE1=True — loading Phase-1 checkpoint, "
              "skipping Phase 1 training.")
        model = load_model(cfg.DL_MODEL_PHASE1_PATH, compile=False)
        model.compile(optimizer=Adam(cfg.LEARNING_RATE), loss=bce,
                      metrics=["accuracy"])
        shutil.copy2(cfg.DL_MODEL_PHASE1_PATH, cfg.DL_MODEL_PATH)
        val_eval    = model.evaluate(val_gen, verbose=0)
        phase1_best = float(val_eval[0])   # val_loss (lower is better)
        print(f"[*] Phase-1 (resumed) val_loss: {phase1_best:.4f}")
        # Dummy history object so the union merge below works cleanly.
        _empty = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
        h1 = type("_H", (), {"history": _empty})()

    else:
        print(f"\n[{cfg.MEMBER_NAME}] Phase 1 — head warm-up  "
              f"({phase1_epochs} epochs, base frozen)")
        model = build_model()
        model.compile(optimizer=Adam(cfg.LEARNING_RATE), loss=bce,
                      metrics=["accuracy"])
        model.summary()

        patience = getattr(cfg, "EARLY_STOPPING_PATIENCE", 6)
        cb1 = [
            _make_checkpoint(initial_best=np.inf, monitor="val_loss"),
            EarlyStopping(monitor="val_loss",
                          patience=patience,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=max(1, patience // 2),
                              min_lr=1e-7, verbose=1),
        ]

        t0 = time.time()
        h1 = model.fit(
            train_gen,
            validation_data = val_gen,
            epochs          = phase1_epochs,
            callbacks       = cb1,
            class_weight    = class_weight,
        )
        elapsed     = time.time() - t0
        total_time += elapsed

        # Save Phase-1 best as a safety snapshot.
        if os.path.exists(cfg.DL_MODEL_PATH):
            shutil.copy2(cfg.DL_MODEL_PATH, cfg.DL_MODEL_PHASE1_PATH)

        _phase_summary("Phase 1 — head warm-up", h1.history, elapsed)
        # Seed Phase 2 checkpoint with Phase 1's best val_loss.
        losses = h1.history.get("val_loss", [])
        phase1_best = min(losses) if losses else np.inf

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 2 — selective base fine-tuning with cosine-decay LR
    # ──────────────────────────────────────────────────────────────────────
    print(f"[{cfg.MEMBER_NAME}] Phase 2 — base fine-tuning  "
          f"({cfg.PHASE2_EPOCHS} epochs, last {cfg.PHASE2_UNFREEZE} base layers)")
    _unfreeze_for_phase2(model)

    phase2_lr = cfg.LEARNING_RATE / 10.0

    # ── CRITICAL FIX ──────────────────────────────────────────────────────
    # Pass a plain float to Adam, NOT a CosineDecay schedule object.
    # On TF 2.10 / Windows, Adam(CosineDecay(...)) stores the schedule object
    # as self.lr (a tf.Variable wrapping a Python object).  When Keras Progbar
    # calls  lr_value * batch_count (int),  the multiplication is dispatched to
    # CosineDecay.__mul__ which is not defined, raising:
    #     TypeError: unsupported operand type(s) for *: 'CosineDecay' and 'int'
    # The LearningRateScheduler callback writes a plain float to optimizer.lr
    # at the start of each epoch — the Progbar then reads a float, not a
    # schedule object, and the multiplication never happens.
    # ──────────────────────────────────────────────────────────────────────
    model.compile(
        optimizer = Adam(learning_rate=phase2_lr),  # plain float — safe on TF 2.10
        loss      = bce,
        metrics   = ["accuracy"],
    )

    patience = getattr(cfg, "EARLY_STOPPING_PATIENCE", 6)
    cb2 = [
        _make_checkpoint(initial_best=phase1_best, monitor="val_loss"),
        EarlyStopping(monitor="val_loss",
                      patience=patience,
                      restore_best_weights=True, verbose=1),
        # Cosine-decay as a callback — returns a plain Python float each epoch.
        _cosine_decay_callback(phase2_lr, cfg.PHASE2_EPOCHS, alpha=0.05),
        # ReduceLROnPlateau intentionally omitted — it fights the cosine decay.
    ]

    # SWA: average weights from the last fraction of Phase 2.
    use_swa = getattr(cfg, "USE_SWA", False)
    if use_swa:
        swa_frac  = getattr(cfg, "SWA_START_FRAC", 0.6)
        swa_start = int(cfg.PHASE2_EPOCHS * swa_frac)
        pre_swa_path = cfg.DL_MODEL_PATH.replace(".h5", "_pre_swa.h5")
        cb2.append(_SWACallback(start_epoch=swa_start,
                                save_pre_swa=pre_swa_path))
        print(f"[*] SWA enabled: averaging from epoch {swa_start}/{cfg.PHASE2_EPOCHS}")

    t0 = time.time()
    h2 = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = cfg.PHASE2_EPOCHS,
        callbacks       = cb2,
        class_weight    = class_weight,
    )
    elapsed     = time.time() - t0
    total_time += elapsed
    phase2_best = _phase_summary("Phase 2 — base fine-tuning",
                                 h2.history, elapsed)

    # If SWA was active, save the SWA-averaged model as the primary checkpoint.
    if use_swa:
        model.save(cfg.DL_MODEL_PATH)
        print(f"[SWA] Saved SWA-averaged model → {cfg.DL_MODEL_PATH}")

    print(f"  ══ Combined training time: {total_time:.0f}s  "
          f"  Best val_acc: {phase2_best:.4f} ══\n")

    # ── History merge (UNION of keys) ──────────────────────────────────────
    # Using  set(h1) | set(h2)  ensures Phase-2-only keys like 'lr' (written
    # by LearningRateScheduler) are not silently dropped.
    all_keys  = set(h1.history) | set(h2.history)
    full_hist = {k: list(h1.history.get(k, [])) + list(h2.history.get(k, []))
                 for k in all_keys}
    full_hist["train_time_sec"]           = total_time
    full_hist["phase1_best_val_accuracy"] = phase1_best
    full_hist["phase2_best_val_accuracy"] = phase2_best
    np.save(cfg.DL_HISTORY_PATH, full_hist)

    plot_training_curves(
        full_hist,
        title     = f"End-to-End {cfg.MODEL_NAME} — {cfg.MEMBER_NAME}",
        save_path = os.path.join(cfg.RESULTS_DIR, "dl_training_curves.png"),
    )

    # ── Reload globally best checkpoint ────────────────────────────────────
    print(f"[*] Reloading best checkpoint: "
          f"{os.path.basename(cfg.DL_MODEL_PATH)}")
    try:
        best_model = load_model(cfg.DL_MODEL_PATH, compile=False)
    except Exception as exc:
        print(f"[!] Checkpoint reload failed ({exc}); using in-memory model.")
        best_model = model

    # ── Inference ──────────────────────────────────────────────────────────
    use_tta = getattr(cfg, "USE_TTA", False)
    if use_tta:
        print("[*] TTA active: averaging over 8 deterministic views "
              "(original + H-flip + V-flip + rot90 + 4 corner crops).")

    # Val probabilities — saved for Step 6 threshold tuning.
    y_val_true, y_val_proba = _predict_all(best_model, val_gen,  use_tta=use_tta)
    np.save(os.path.join(cfg.RESULTS_DIR, "dl_val_proba.npy"),
            np.column_stack([y_val_true, y_val_proba]))

    # Test probabilities — final evaluation.
    y_true, y_proba = _predict_all(best_model, test_gen, use_tta=use_tta)
    np.save(os.path.join(cfg.RESULTS_DIR, "dl_test_proba.npy"),
            np.column_stack([y_true, y_proba]))

    y_pred   = (y_proba > 0.5).astype(int)
    m        = compute_metrics(y_true, y_pred,
                               model_name=f"End-to-End {cfg.MODEL_NAME}")
    m_proba  = compute_metrics_with_proba(y_true, y_proba,
                                          model_name=f"End-to-End {cfg.MODEL_NAME} (proba)")
    m.update({k: v for k, v in m_proba.items() if k not in m})
    m["train_time_sec"] = total_time

    # ── Youden's J threshold (tuned on val) ────────────────────────────
    # Youden's J = TPR − FPR.  Often picks a more conservative threshold
    # that generalises better when test distribution differs from val.
    from utils.metrics import find_best_threshold
    t_youden, _ = find_best_threshold(y_val_true, y_val_proba,
                                       pos_label=1, criterion="youden")
    y_pred_youden = (y_proba >= t_youden).astype(int)
    m_youden = compute_metrics(y_true, y_pred_youden,
                               model_name=f"End-to-End {cfg.MODEL_NAME} "
                                          f"@ Youden thr={t_youden:.2f}")
    m["youden_threshold"] = float(t_youden)
    for k, v in m_youden.items():
        m[f"youden_{k}"] = v
    print(f"  Youden's J threshold (val-tuned): {t_youden:.2f}")

    np.save(os.path.join(cfg.RESULTS_DIR, "dl_results.npy"), m)

    plot_confusion_matrix(
        y_true, y_pred, cfg.CLASS_NAMES,
        title     = f"End-to-End Confusion Matrix — {cfg.MEMBER_NAME}",
        save_path = os.path.join(cfg.RESULTS_DIR, "cm_dl.png"),
    )

    print(f"\n[✓] End-to-End DL complete.  Results → {cfg.RESULTS_DIR}\n")


if __name__ == "__main__":
    run()
