"""
config.py — Mario's standalone config. No shared dependencies.
Dataset : PlantVillage (Leaf Disease)  |  Architecture: ResNet50
"""
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

MEMBER_NAME  = "Mario_plantvillage"
DATASET_NAME = "PlantVillage (Leaf Disease)"
MODEL_NAME   = "ResNet50"

HERE = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT  = os.path.join(HERE, "data", "plant_village")
TRAIN_DIR  = os.path.join(DATA_ROOT, "train")
VAL_DIR    = os.path.join(DATA_ROOT, "val")
TEST_DIR   = os.path.join(DATA_ROOT, "test")

CLASS_NAMES = None        # inferred from directory (38 disease classes)
NUM_CLASSES = 38
CLASS_MODE  = "categorical"

MODEL_BUILDER = ResNet50
PREPROCESS_FN = preprocess_input

IMG_SIZE    = (224, 224)
INPUT_SHAPE = (224, 224, 3)

AUGMENTATION = dict(
    rotation_range      = 30,
    width_shift_range   = 0.15,
    height_shift_range  = 0.15,
    zoom_range          = 0.2,
    horizontal_flip     = True,
    vertical_flip       = True,
    brightness_range    = [0.8, 1.2],
    fill_mode           = "nearest",
)

BATCH_SIZE    = 32
EPOCHS        = 20
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
DL_MODEL_PATH   = os.path.join(MODELS_DIR, "resnet50_endtoend.h5")
DL_HISTORY_PATH = os.path.join(RESULTS_DIR, "dl_history.npy")
