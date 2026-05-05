"""
members/friend_b_cifar10/config.py
────────────────────────────────────────────────────────────────────────────────
Friend B — CIFAR-10 (auto-downloaded) + ResNet50.
CIFAR-10 images are 32x32, resized to 224x224 inside data_loader.py.
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.base_config import *
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# ── Identity ──────────────────────────────────────────────────────────────────
MEMBER_NAME  = "friend_b_cifar10"   # matches folder name exactly
DATASET_NAME = "CIFAR-10"
MODEL_NAME   = "ResNet50"

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_MODE   = "builtin"
DATASET_LOADER = tf.keras.datasets.cifar10.load_data

CLASS_NAMES = ["airplane","automobile","bird","cat","deer",
               "dog","frog","horse","ship","truck"]
NUM_CLASSES = 10
CLASS_MODE  = "categorical"

IMG_SIZE    = (224, 224)
INPUT_SHAPE = (224, 224, 3)

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_BUILDER = ResNet50
PREPROCESS_FN = preprocess_input

EPOCHS = 20

# ── Output paths ───────────────────────────────────────────────────────────────
_ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESULTS_DIR  = os.path.join(_ROOT, "results",  MEMBER_NAME)
MODELS_DIR   = os.path.join(_ROOT, "models",   MEMBER_NAME)
FEATURES_DIR = os.path.join(MODELS_DIR, "features")

for _d in [RESULTS_DIR, MODELS_DIR, FEATURES_DIR]:
    os.makedirs(_d, exist_ok=True)

FEATURES_TRAIN  = os.path.join(FEATURES_DIR, "train_features.npy")
LABELS_TRAIN    = os.path.join(FEATURES_DIR, "train_labels.npy")
FEATURES_VAL    = os.path.join(FEATURES_DIR, "val_features.npy")
LABELS_VAL      = os.path.join(FEATURES_DIR, "val_labels.npy")
FEATURES_TEST   = os.path.join(FEATURES_DIR, "test_features.npy")
LABELS_TEST     = os.path.join(FEATURES_DIR, "test_labels.npy")

DL_MODEL_PATH   = os.path.join(MODELS_DIR, "resnet50_cifar10_endtoend.h5")
DL_HISTORY_PATH = os.path.join(RESULTS_DIR, "dl_history.npy")
