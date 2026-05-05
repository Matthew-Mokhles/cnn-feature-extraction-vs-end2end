"""
members/mark_xray/config.py  — Mark's standalone config. No shared dependencies.
Dataset : Chest X-Ray (Pneumonia)  |  Architecture: VGG16
"""
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

MEMBER_NAME  = "mark_xray"
DATASET_NAME = "Chest X-Ray (Pneumonia)"
MODEL_NAME   = "VGG16"

HERE = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT  = os.path.join(HERE, "data", "chest_xray")
TRAIN_DIR  = os.path.join(DATA_ROOT, "train")
VAL_DIR    = os.path.join(DATA_ROOT, "val")
TEST_DIR   = os.path.join(DATA_ROOT, "test")

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES = 2
CLASS_MODE  = "binary"

MODEL_BUILDER = VGG16
PREPROCESS_FN = preprocess_input

IMG_SIZE    = (224, 224)
INPUT_SHAPE = (224, 224, 3)

AUGMENTATION = dict(
    rotation_range     = 15,
    width_shift_range  = 0.1,
    height_shift_range = 0.1,
    zoom_range         = 0.1,
    horizontal_flip    = True,
    fill_mode          = "nearest",
)

BATCH_SIZE    = 32
EPOCHS        = 15
LEARNING_RATE = 1e-4
DROPOUT_RATE  = 0.5
RANDOM_SEED   = 42

SVM_KERNEL  = "linear"
SVM_C       = 1.0
LR_C        = 1.0
LR_MAX_ITER = 1000

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
DL_MODEL_PATH   = os.path.join(MODELS_DIR, "vgg16_endtoend.h5")
DL_HISTORY_PATH = os.path.join(RESULTS_DIR, "dl_history.npy")
