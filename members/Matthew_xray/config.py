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
AUGMENTATION = dict(
    rotation_range     = 15,
    width_shift_range  = 0.1,
    height_shift_range = 0.1,
    zoom_range         = 0.1,
    horizontal_flip    = True,
    fill_mode          = "nearest",
)

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 15
LEARNING_RATE = 1e-4
DROPOUT_RATE  = 0.5
RANDOM_SEED   = 42

# Validation split quality — used by preprocessing / endtoend_dl warnings.
# Aim for hundreds per class (e.g. 10–20% of train); tiny val makes
# val_accuracy / ModelCheckpoint metrics very noisy (see README).
MIN_RECOMMENDED_VAL_SAMPLES = 100

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

SVM_MODEL_PATH  = os.path.join(MODELS_DIR, "svm_classifier.joblib")
LR_MODEL_PATH   = os.path.join(MODELS_DIR, "lr_classifier.joblib")
DL_MODEL_PATH   = os.path.join(MODELS_DIR, "resnet50_endtoend.h5")
DL_HISTORY_PATH = os.path.join(RESULTS_DIR, "dl_history.npy")
