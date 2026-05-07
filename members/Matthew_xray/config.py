"""
config.py — Matthew's standalone configuration. No shared dependencies.
All paths are relative to this file's location (members/Matthew_xray/).
"""
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# ── Identity ──────────────────────────────────────────────────────────────────
MEMBER_NAME  = "Matthew_xray"
DATASET_NAME = "Chest X-Ray (Pneumonia)"
MODEL_NAME   = "ResNet50"

# ── Root of this workspace ────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))

# ── Dataset — place data at  members/Matthew_xray/data/chest_xray/ ───────────
DATA_ROOT  = os.path.join(HERE, "data", "chest_xray")
TRAIN_DIR  = os.path.join(DATA_ROOT, "train")
VAL_DIR    = os.path.join(DATA_ROOT, "val")
TEST_DIR   = os.path.join(DATA_ROOT, "test")

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES = 2
CLASS_MODE  = "binary"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_BUILDER = ResNet50
PREPROCESS_FN = preprocess_input

# ── Image settings ────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
INPUT_SHAPE = (224, 224, 3)

# ── Augmentation ──────────────────────────────────────────────────────────────
# Aggressive augmentation for chest x-rays — designed to close the domain gap
# between train/val (~74% PNEUMONIA) and test (~63% PNEUMONIA) distributions.
# brightness_range, shear_range, and larger shifts simulate different
# acquisition setups and patient positioning.
AUGMENTATION = dict(
    rotation_range     = 20,
    width_shift_range  = 0.15,
    height_shift_range = 0.15,
    zoom_range         = 0.15,
    shear_range        = 0.08,
    brightness_range   = (0.80, 1.20),
    horizontal_flip    = True,
    fill_mode          = "nearest",
)

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 25
LEARNING_RATE = 1e-4
DROPOUT_RATE  = 0.4
RANDOM_SEED   = 42

# Validation split quality — used by preprocessing / endtoend_dl warnings.
# Aim for hundreds per class (e.g. 10–20% of train); tiny val makes
# val_accuracy / ModelCheckpoint metrics very noisy (see README).
MIN_RECOMMENDED_VAL_SAMPLES = 100

# ── DL training (end-to-end) ─────────────────────────────────────────────────
# Phase 2 fine-tunes the last PHASE2_UNFREEZE layers of the base for
# PHASE2_EPOCHS epochs with a cosine-decayed LR. Label smoothing regularises
# the binary cross-entropy loss.
LABEL_SMOOTHING       = 0.1
PHASE2_EPOCHS         = 15
PHASE2_UNFREEZE       = 15
PHASE2_FREEZE_BN      = True
USE_TF_DATA           = True
EARLY_STOPPING_PATIENCE = 6

# ── SWA (Stochastic Weight Averaging) ─────────────────────────────────────────
# Averages model weights from the last fraction of Phase 2 training to find a
# flatter minimum that generalises better to the shifted test distribution.
USE_SWA               = True
SWA_START_FRAC        = 0.6      # start averaging from 60% of Phase 2

# ── Test-Time Augmentation ────────────────────────────────────────────────────
USE_TTA               = True

# ── ML baselines (Step 4) ────────────────────────────────────────────────────
# CV scoring is **f1_macro** so the GridSearch is penalised when a candidate
# ignores NORMAL (the minority class). The grid was tightened after the first
# run showed that C=10 + RBF + L2-normalised features collapsed into a
# "predict PNEUMONIA everywhere" rule on the held-out test set even though
# train+val CV looked near-perfect.
ML_KFOLDS                 = 5
ML_SCORING                = "f1_macro"
ML_GRID_SVM_C             = [0.01, 0.1, 1.0]
ML_GRID_SVM_KERNELS       = ["linear"]
ML_GRID_SVM_CLASS_WEIGHT  = ["balanced", {0: 2.0, 1: 1.0}]
ML_GRID_LR_C              = [0.01, 0.1, 1.0]
ML_GRID_LR_CLASS_WEIGHT   = ["balanced", {0: 2.0, 1: 1.0}]

# Legacy single-shot params (kept for backwards-compat / debugging).
SVM_KERNEL  = "linear"
SVM_C       = 1.0
LR_C        = 1.0
LR_MAX_ITER = 1000

# ── Output paths (all inside members/Matthew_xray/) ────────────────────────────
RESULTS_DIR  = os.path.join(HERE, "results")
MODELS_DIR   = os.path.join(HERE, "models")
FEATURES_DIR = os.path.join(MODELS_DIR, "features")

for _d in [RESULTS_DIR, MODELS_DIR, FEATURES_DIR]:
    os.makedirs(_d, exist_ok=True)

FEATURES_TRAIN  = os.path.join(FEATURES_DIR, "train_features.npy")
LABELS_TRAIN    = os.path.join(FEATURES_DIR, "train_labels.npy")
FEATURES_VAL    = os.path.join(FEATURES_DIR, "val_features.npy")
LABELS_VAL      = os.path.join(FEATURES_DIR, "val_labels.npy")
FEATURES_TEST   = os.path.join(FEATURES_DIR, "test_features.npy")
LABELS_TEST     = os.path.join(FEATURES_DIR, "test_labels.npy")

SVM_MODEL_PATH       = os.path.join(MODELS_DIR, "svm_classifier.joblib")
LR_MODEL_PATH        = os.path.join(MODELS_DIR, "lr_classifier.joblib")
MLP_MODEL_PATH       = os.path.join(MODELS_DIR, "mlp_classifier.joblib")
DL_MODEL_PATH        = os.path.join(MODELS_DIR, "resnet50_endtoend.h5")
DL_MODEL_PHASE1_PATH = os.path.join(MODELS_DIR, "resnet50_endtoend_phase1.h5")
DL_HISTORY_PATH      = os.path.join(RESULTS_DIR, "dl_history.npy")
