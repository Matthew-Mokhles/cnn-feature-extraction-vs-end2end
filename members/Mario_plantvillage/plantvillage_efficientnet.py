"""
=============================================================================
PlantVillage Dataset - EfficientNet-B0 Comparative Study
=============================================================================
Course  : CAI3105/CS460 – Deep Learning
Project : 12th Week Project
Dataset : PlantVillage (Kaggle: abdallahalidev/plantvillage-dataset)
Model   : EfficientNet-B0

Approach 1 : End-to-End EfficientNet-B0 (Fine-tuned Deep Learning classifier)
Approach 2 : EfficientNet-B0 as Fixed Feature Extractor + SVM Classifier

Requirements covered:
  Req-1 : Dataset selection, metadata, preprocessing, augmentation, splits
  Req-2 : DL model selection + hyperparameter table
  Req-3 : Implementation of both approaches with metrics
  Req-4 : Comparative analysis with plots and conclusion
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────────────────────────────────────
import os, time, warnings, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
print(f"TensorFlow version : {tf.__version__}")
print(f"GPU available      : {tf.config.list_physical_devices('GPU')}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION  (Requirement 1 & 2 parameters)
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # ── Dataset ──────────────────────────────────────────────────────────────
    "dataset_path"   : "D:\\semster 8\\Deep Learning\\assignments\\12th_project\\plantvillage dataset",        # path after unzip
    "img_size"       : (224, 224),
    "color_channels" : 3,                        # RGB
    "batch_size"     : 32,

    # ── Splits ───────────────────────────────────────────────────────────────
    "train_ratio"    : 0.70,
    "val_ratio"      : 0.15,
    "test_ratio"     : 0.15,
    "random_seed"    : 42,

    # ── Approach-1 (End-to-End DL) ───────────────────────────────────────────
    "dl_epochs"      : 20,
    "dl_lr"          : 1e-4,
    "dl_dropout"     : 0.5,
    "dl_optimizer"   : "Adam",
    "dl_loss"        : "categorical_crossentropy",
    "fine_tune_at"   : 100,                      # unfreeze from layer 100 onward

    # ── Approach-2 (Feature Extractor + SVM) ─────────────────────────────────
    "svm_kernel"     : "rbf",
    "svm_C"          : 10.0,
    "svm_gamma"      : "scale",
    "pca_components" : 256,                      # reduce 1280-dim features with PCA
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING & PREPROCESSING  (Requirement 1)
# ─────────────────────────────────────────────────────────────────────────────
def build_data_generators(dataset_path, img_size, batch_size, seed=42):
    """
    Builds train / validation / test ImageDataGenerators.

    Preprocessing  : EfficientNet-specific pixel normalisation (preprocess_input).
    Augmentation   : Rotation ±20°, width/height shift ±15%, horizontal flip,
                     zoom ±15%, shear ±10°.
    Justification  : PlantVillage images are controlled lab shots; augmentation
                     forces the model to learn disease features (lesion shape,
                     colour) rather than fixed backgrounds, thus improving
                     real-world generalisation.
    """
    # ── Augmented generator for training ─────────────────────────────────────
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15,
        shear_range=10,
        validation_split=CFG["val_ratio"] / (CFG["train_ratio"] + CFG["val_ratio"]),
    )

    # ── No-augmentation generator for val/test ────────────────────────────────
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    color_dir = os.path.join(dataset_path, "color")
    if not os.path.exists(color_dir):
        color_dir = dataset_path          # fallback if already at class level

    train_gen = train_datagen.flow_from_directory(
        color_dir, target_size=img_size, batch_size=batch_size,
        class_mode="categorical", subset="training", seed=seed,
        shuffle=True,
    )
    val_gen = train_datagen.flow_from_directory(
        color_dir, target_size=img_size, batch_size=batch_size,
        class_mode="categorical", subset="validation", seed=seed,
        shuffle=False,
    )
    test_gen = test_datagen.flow_from_directory(
        color_dir, target_size=img_size, batch_size=batch_size,
        class_mode="categorical", seed=seed, shuffle=False,
    )

    return train_gen, val_gen, test_gen

# ─────────────────────────────────────────────────────────────────────────────
# 3. APPROACH-1 : END-TO-END EFFICIENTNET-B0  (Requirement 2 & 3)
# ─────────────────────────────────────────────────────────────────────────────
def build_e2e_model(num_classes: int) -> Model:
    """
    EfficientNet-B0 fine-tuned end-to-end model.
    Architecture:
      EfficientNetB0 (ImageNet weights, top removed)
      → GlobalAveragePooling2D
      → BatchNormalization
      → Dropout(0.5)
      → Dense(num_classes, softmax)
    """
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(*CFG["img_size"], CFG["color_channels"]),
    )
    # Phase-1: freeze entire base, train new head
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(CFG["dl_dropout"])(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)
    return model, base


def train_e2e(train_gen, val_gen, num_classes):
    print("\n" + "="*60)
    print("  APPROACH-1 : END-TO-END EFFICIENTNET-B0")
    print("="*60)

    model, base = build_e2e_model(num_classes)

    # ── Phase-1: train head only ──────────────────────────────────────────────
    model.compile(
        optimizer=Adam(CFG["dl_lr"]),
        loss=CFG["dl_loss"],
        metrics=["accuracy"],
    )
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ]
    print("\n[Phase-1] Training classification head …")
    h1 = model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks, verbose=1)

    # ── Phase-2: fine-tune upper layers of base ───────────────────────────────
    base.trainable = True
    for layer in base.layers[:CFG["fine_tune_at"]]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(CFG["dl_lr"] / 10),
        loss=CFG["dl_loss"],
        metrics=["accuracy"],
    )
    print(f"\n[Phase-2] Fine-tuning from layer {CFG['fine_tune_at']} onward …")
    h2 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=CFG["dl_epochs"], callbacks=callbacks, verbose=1,
    )

    # Merge histories
    history = {
        "loss"         : h1.history["loss"]          + h2.history["loss"],
        "val_loss"     : h1.history["val_loss"]       + h2.history["val_loss"],
        "accuracy"     : h1.history["accuracy"]       + h2.history["accuracy"],
        "val_accuracy" : h1.history["val_accuracy"]   + h2.history["val_accuracy"],
    }
    return model, history


def evaluate_dl_model(model, test_gen):
    print("\n[Approach-1] Evaluating on test set …")
    t0 = time.time()
    y_pred_prob = model.predict(test_gen, verbose=1)
    inference_time = time.time() - t0

    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_gen.classes
    classes = list(test_gen.class_indices.keys())

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
               "inference_time": inference_time, "confusion_matrix": cm, "classes": classes}
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 4. APPROACH-2 : EFFICIENTNET-B0 FEATURE EXTRACTOR + SVM  (Req 2 & 3)
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_extractor():
    """Returns EfficientNetB0 without top (fixed weights → feature extractor)."""
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(*CFG["img_size"], CFG["color_channels"]),
    )
    base.trainable = False
    return base


def extract_features(feature_extractor, generator):
    """Passes all batches through the frozen base to get 1280-D feature vectors."""
    print("  Extracting features …", end="", flush=True)
    generator.reset()
    features, labels = [], []
    for i in range(len(generator)):
        x_batch, y_batch = generator[i]
        feat = feature_extractor.predict(x_batch, verbose=0)
        features.append(feat)
        labels.append(np.argmax(y_batch, axis=1))
        print(f"\r  Extracting features … batch {i+1}/{len(generator)}", end="", flush=True)
    print()
    return np.vstack(features), np.concatenate(labels)


def train_svm(X_train, y_train, X_val, y_val):
    print("\n" + "="*60)
    print("  APPROACH-2 : EFFICIENTNET-B0 + SVM")
    print("="*60)

    # PCA to compress 1280 → 256 dims (speeds up SVM without big accuracy loss)
    print(f"\n[PCA] Reducing {X_train.shape[1]}-D → {CFG['pca_components']}-D …")
    pca = PCA(n_components=CFG["pca_components"], random_state=CFG["random_seed"])
    X_train_r = pca.fit_transform(X_train)
    X_val_r   = pca.transform(X_val)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_r)
    X_val_s   = scaler.transform(X_val_r)

    print(f"[SVM] Training RBF-SVM (C={CFG['svm_C']}, gamma={CFG['svm_gamma']}) …")
    t0 = time.time()
    svm = SVC(
        kernel=CFG["svm_kernel"], C=CFG["svm_C"],
        gamma=CFG["svm_gamma"], decision_function_shape="ovr",
        random_state=CFG["random_seed"], verbose=False,
    )
    svm.fit(X_train_s, y_train)
    train_time = time.time() - t0
    print(f"  SVM training time: {train_time:.1f}s")
    return svm, pca, scaler, train_time


def evaluate_svm(svm, pca, scaler, X_test, y_test, classes):
    print("\n[Approach-2] Evaluating SVM on test set …")
    X_test_r = pca.transform(X_test)
    X_test_s = scaler.transform(X_test_r)

    t0 = time.time()
    y_pred = svm.predict(X_test_s)
    inference_time = time.time() - t0

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
               "inference_time": inference_time, "confusion_matrix": cm, "classes": classes}
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATIONS  (Requirement 4)
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {"dl": "#2196F3", "svm": "#FF5722"}

def plot_learning_curves(history, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Approach-1 : EfficientNet-B0 End-to-End – Learning Curves", fontsize=14, fontweight="bold")

    for ax, metric, title in zip(axes,
                                  [("accuracy","val_accuracy"), ("loss","val_loss")],
                                  ["Accuracy", "Loss"]):
        ax.plot(history[metric[0]], label="Train", color=COLORS["dl"], linewidth=2)
        ax.plot(history[metric[1]], label="Validation", color="#64B5F6", linewidth=2, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_metric_comparison(m_dl, m_svm, save_dir):
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, [m_dl[m]  for m in metrics], width,
                   label="Approach-1: End-to-End DL", color=COLORS["dl"])
    bars2 = ax.bar(x + width/2, [m_svm[m] for m in metrics], width,
                   label="Approach-2: DL Features + SVM", color=COLORS["svm"])

    ax.set_title("Performance Comparison: End-to-End DL vs DL + SVM", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in [*bars1, *bars2]:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, "metric_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_confusion_matrix(cm, classes, title, fname, save_dir, max_classes=20):
    """Plot confusion matrix; if too many classes, show top-N."""
    if len(classes) > max_classes:
        # Keep only top-N most confused classes for readability
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        idx = np.argsort(per_class_acc)[:max_classes]
        cm = cm[np.ix_(idx, idx)]
        classes = [classes[i] for i in idx]
        title += f" (top-{max_classes} lowest-accuracy classes)"

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=len(classes) <= 15, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.3, linecolor="gray")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    path = os.path.join(save_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_class_distribution(test_gen, save_dir):
    classes  = list(test_gen.class_indices.keys())
    counts   = np.bincount(test_gen.classes)
    idx_sort = np.argsort(counts)[::-1][:30]          # top-30 by count

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar([classes[i] for i in idx_sort], counts[idx_sort], color="#4CAF50")
    ax.set_title("Class Distribution in Dataset (top 30 classes)", fontweight="bold")
    ax.set_ylabel("Number of Samples")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    path = os.path.join(save_dir, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_dataset_overview(save_dir):
    """Infographic-style overview of dataset metadata."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    data = [
        ["Dataset",    "PlantVillage (Color subset)"],
        ["Source",     "Kaggle / Penn State University"],
        ["Total Images","~55,448"],
        ["Classes",    "38 (14 crop species)"],
        ["Resolution", "224 × 224 px (RGB)"],
        ["Train/Val/Test", "70% / 15% / 15%"],
        ["Problem",    "Multi-class plant disease classification"],
    ]
    table = ax.table(cellText=data, colLabels=["Property","Value"],
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#E3F2FD")
    ax.set_title("Dataset Metadata  (Requirement 1)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    path = os.path.join(save_dir, "dataset_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_hyperparameter_table(save_dir):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")
    data = [
        ["Pre-trained backbone", "EfficientNet-B0 (ImageNet weights)", "EfficientNet-B0 (frozen)"],
        ["Classifier head",      "Dense(num_classes, softmax)",        "RBF-SVM"],
        ["Feature dimension",    "1280 → classes",                    "1280 → PCA(256) → SVM"],
        ["Learning rate",        "1e-4 (head) / 1e-5 (fine-tune)",   "—"],
        ["Optimizer",            "Adam",                              "—"],
        ["Epochs",               f"{CFG['dl_epochs']} (+ 10 warmup)", "—"],
        ["Batch size",           str(CFG["batch_size"]),              str(CFG["batch_size"])],
        ["SVM kernel",           "—",                                 CFG["svm_kernel"].upper()],
        ["SVM C",                "—",                                 str(CFG["svm_C"])],
        ["SVM gamma",            "—",                                 CFG["svm_gamma"]],
        ["PCA components",       "—",                                 str(CFG["pca_components"])],
        ["Dropout",              str(CFG["dl_dropout"]),              "—"],
        ["Loss function",        CFG["dl_loss"],                     "—"],
        ["Early stopping",       "patience=5",                        "—"],
    ]
    col_labels = ["Hyperparameter", "Approach-1 (End-to-End DL)", "Approach-2 (DL + SVM)"]
    table = ax.table(cellText=data, colLabels=col_labels, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#0D47A1")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#E8F5E9")
    ax.set_title("Hyperparameter Summary  (Requirement 2)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(save_dir, "hyperparameter_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)

    dataset_path = CFG["dataset_path"]
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'.\n"
            "Download from: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset\n"
            "Unzip and set dataset_path to the folder containing 'color/', 'grayscale/', 'segmented/'."
        )

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n[1] Building data generators …")
    train_gen, val_gen, test_gen = build_data_generators(
        dataset_path, CFG["img_size"], CFG["batch_size"], CFG["random_seed"]
    )
    num_classes = train_gen.num_classes
    classes     = list(test_gen.class_indices.keys())
    print(f"  Classes  : {num_classes}")
    print(f"  Train    : {train_gen.samples} images")
    print(f"  Val      : {val_gen.samples} images")
    print(f"  Test     : {test_gen.samples} images")

    # ── Static plots ─────────────────────────────────────────────────────────
    plot_dataset_overview(save_dir)
    plot_hyperparameter_table(save_dir)
    plot_class_distribution(test_gen, save_dir)

    # ════════════════════════════════════════════════════════════════════════
    # APPROACH-1 : End-to-End DL
    # ════════════════════════════════════════════════════════════════════════
    t_dl_start = time.time()
    dl_model, history = train_e2e(train_gen, val_gen, num_classes)
    dl_train_time = time.time() - t_dl_start

    dl_metrics = evaluate_dl_model(dl_model, test_gen)
    dl_metrics["train_time"] = dl_train_time

    plot_learning_curves(history, save_dir)
    plot_confusion_matrix(
        dl_metrics["confusion_matrix"], classes,
        "Approach-1: End-to-End EfficientNet-B0 – Confusion Matrix",
        "cm_dl.png", save_dir
    )

    # ════════════════════════════════════════════════════════════════════════
    # APPROACH-2 : Feature Extractor + SVM
    # ════════════════════════════════════════════════════════════════════════
    print("\n[2] Extracting features for SVM …")
    feature_extractor = build_feature_extractor()

    X_train, y_train = extract_features(feature_extractor, train_gen)
    X_val,   y_val   = extract_features(feature_extractor, val_gen)
    X_test,  y_test  = extract_features(feature_extractor, test_gen)

    t_svm_start = time.time()
    svm_model, pca, scaler, _ = train_svm(X_train, y_train, X_val, y_val)
    svm_train_time = time.time() - t_svm_start

    svm_metrics = evaluate_svm(svm_model, pca, scaler, X_test, y_test, classes)
    svm_metrics["train_time"] = svm_train_time

    plot_confusion_matrix(
        svm_metrics["confusion_matrix"], classes,
        "Approach-2: EfficientNet-B0 Features + SVM – Confusion Matrix",
        "cm_svm.png", save_dir
    )

    # ── Comparison chart ─────────────────────────────────────────────────────
    plot_metric_comparison(dl_metrics, svm_metrics, save_dir)

    # ── Training-time bar chart ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(["Approach-1\n(End-to-End DL)", "Approach-2\n(DL + SVM)"],
                  [dl_train_time / 60, svm_train_time / 60],
                  color=[COLORS["dl"], COLORS["svm"]], width=0.5)
    ax.set_ylabel("Training Time (minutes)")
    ax.set_title("Training Time Comparison", fontweight="bold")
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2,
                f"{b.get_height():.1f} min", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_time.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── Print final summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)
    for name, m in [("Approach-1 (End-to-End DL)", dl_metrics),
                     ("Approach-2 (DL + SVM)",      svm_metrics)]:
        print(f"\n  {name}")
        print(f"    Accuracy  : {m['accuracy']:.4f}")
        print(f"    Precision : {m['precision']:.4f}")
        print(f"    Recall    : {m['recall']:.4f}")
        print(f"    F1-Score  : {m['f1']:.4f}")
        print(f"    Train Time: {m['train_time']/60:.1f} min")

    # Save metrics JSON
    results = {
        "dl" : {k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in dl_metrics.items()  if k != "confusion_matrix"},
        "svm": {k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in svm_metrics.items() if k != "confusion_matrix"},
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nAll results saved to: {save_dir}/")
    return dl_metrics, svm_metrics, history


if __name__ == "__main__":
    main()
