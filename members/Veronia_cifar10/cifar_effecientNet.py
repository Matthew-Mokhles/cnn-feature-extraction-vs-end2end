
#  IMPORTS
# ================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
# ================================================================
#  GPU / CPU CHECK
# ================================================================
if torch.cuda.is_available():
    device   = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[GPU] Running on: {gpu_name}  |  VRAM: {vram:.1f} GB")
else:
    device = torch.device("cpu")
    print("[CPU] No GPU detected — running on CPU (slower).")

print(f"      Active device: {device}\n")


# ================================================================
#  REQUIREMENT 1 — DATASET SELECTION & TECHNICAL SPECIFICATIONS
# ================================================================

# ── Hyperparameters (reported in the project table — Requirement 2) ──
IMG_SIZE     = 224      # EfficientNet-B0 expects 224×224 input
BATCH_SIZE   = 64       # number of images per batch
NUM_WORKERS  = 0      # parallel threads for data loading
DATA_DIR     = "./data" # directory where CIFAR-10 will be downloaded
VAL_SPLIT    = 0.1      # 10% of training set used as validation

# End-to-End training hyperparameters (Approach 2)
NUM_EPOCHS    = 10      # number of full passes over the training set
LEARNING_RATE = 1e-4    # small LR since the model is pretrained
WEIGHT_DECAY  = 1e-4    # L2 regularization to reduce overfitting

# SVM hyperparameters (Approach 1)
SVM_KERNEL   = "linear" # as specified in Requirement 2
SVM_C        = 1.0      # regularization — higher C = less tolerance for errors
SVM_MAX_ITER = 2000     # maximum number of iterations for SVM solver

# ImageNet normalization statistics
#   Used because EfficientNet-B0 was originally trained on ImageNet.
#   Applying the same normalization ensures the input distribution
#   matches what the model expects.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Data Augmentation & Preprocessing ────────────────────────────────
# ─────────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# ── Load CIFAR-10 ─────────────────────────────────────────────────────
print("=" * 60)
print("  REQUIREMENT 1 — Loading CIFAR-10 Dataset")
print("=" * 60)

full_train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True,
                                      download=True, transform=train_transform)
test_dataset       = datasets.CIFAR10(root=DATA_DIR, train=False,
                                      download=True, transform=val_test_transform)

# Split training set into train (90%) and validation (10%)
val_size   = int(len(full_train_dataset) * VAL_SPLIT)   # 5,000 images
train_size = len(full_train_dataset) - val_size          # 45,000 images

train_dataset, val_dataset = random_split(
    full_train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # fixed seed for reproducibility
)

CLASS_NAMES = full_train_dataset.classes
print(f"  Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
print(f"  Training   samples : {len(train_dataset):,}")
print(f"  Validation samples : {len(val_dataset):,}")
print(f"  Test       samples : {len(test_dataset):,}\n")

# pin_memory=True speeds up CPU→GPU transfers when a GPU is available
pin = torch.cuda.is_available()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=NUM_WORKERS, pin_memory=pin)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

# A separate loader for feature extraction (shuffle=False to keep label order)
extract_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)


# ================================================================
#  REQUIREMENT 2 — DL MODEL SELECTION
# ================================================================

print("=" * 60)
print("  REQUIREMENT 2 — Model: EfficientNet-B0")
print("=" * 60)
print("  Pretrained on ImageNet | Feature output: 1280-dim\n")

# ================================================================
#  REQUIREMENT 3 — IMPLEMENTATION
# ================================================================
print("=" * 60)
print("  REQUIREMENT 3 — Approach 1: EfficientNet + SVM")
print("=" * 60)

# Load EfficientNet-B0 and strip the classifier head
efficientnet_extractor = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
)
# Identity replaces the classifier so the model outputs raw features (1280-dim)
efficientnet_extractor.classifier = nn.Identity()

# Freeze every parameter — we are using EfficientNet as a fixed feature extractor
for param in efficientnet_extractor.parameters():
    param.requires_grad = False

efficientnet_extractor = efficientnet_extractor.to(device)
efficientnet_extractor.eval()
print("  EfficientNet-B0 loaded (frozen) | Output feature size: 1280\n")


def extract_features(loader, model, device, split_name=""):
    """
    Passes all images in `loader` through the frozen EfficientNet
    and returns the extracted feature matrix and corresponding labels.

    Args:
        loader      : DataLoader (shuffle=False)
        model       : frozen EfficientNet with Identity classifier
        device      : torch.device (cuda or cpu)
        split_name  : string label for logging (e.g. "Training set")

    Returns:
        features : np.ndarray of shape (N, 1280)
        labels   : np.ndarray of shape (N,)
    """
    all_features, all_labels = [], []
    print(f"  Extracting features from {split_name}...")

    with torch.no_grad():   # disable gradient computation — not needed for inference
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            feats  = model(images)              # forward pass → (batch, 1280)
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
            if (i + 1) % 100 == 0:
                print(f"    Batch {i+1}/{len(loader)} done...")

    features = np.concatenate(all_features, axis=0)  # (N, 1280)
    labels   = np.concatenate(all_labels,   axis=0)  # (N,)
    print(f"    Feature matrix shape: {features.shape}\n")
    return features, labels


# Extract features from training and test sets
start_extract = time.time()
X_train, y_train = extract_features(extract_train_loader, efficientnet_extractor,
                                    device, "Training set")
X_test,  y_test  = extract_features(test_loader, efficientnet_extractor,
                                    device, "Test set")
extract_time = time.time() - start_extract
print(f"  Feature extraction completed in {extract_time/60:.1f} min\n")

# Standardize features — zero mean, unit variance
# fit_transform on train only; transform (no fit) on test to prevent data leakage
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print("  StandardScaler applied (fit on train, transform on test)\n")

# Train SVM with a linear kernel
print(f"  Training SVM  (kernel={SVM_KERNEL}, C={SVM_C})...")
svm = SVC(kernel=SVM_KERNEL, C=SVM_C, max_iter=SVM_MAX_ITER,
          random_state=42, verbose=False)

start_svm = time.time()
svm.fit(X_train, y_train)
svm_time  = time.time() - start_svm
print(f"  SVM training completed in {svm_time/60:.1f} min\n")

# ── Approach 1 Evaluation ─────────────────────────────────────────────
y_pred_app1 = svm.predict(X_test)

acc1  = accuracy_score(y_test, y_pred_app1)
prec1 = precision_score(y_test, y_pred_app1, average="weighted", zero_division=0)
rec1  = recall_score(y_test, y_pred_app1, average="weighted", zero_division=0)
f1_1  = f1_score(y_test, y_pred_app1, average="weighted", zero_division=0)
cm1   = confusion_matrix(y_test, y_pred_app1)
time1 = (extract_time + svm_time) / 60

print("=" * 55)
print("  RESULTS — Approach 1 (EfficientNet-B0 + SVM)")
print("=" * 55)
print(f"  Accuracy  : {acc1 * 100:.2f}%")
print(f"  Precision : {prec1 * 100:.2f}%")
print(f"  Recall    : {rec1 * 100:.2f}%")
print(f"  F1-Score  : {f1_1 * 100:.2f}%")
print(f"  Total Time: {time1:.1f} min  "
      f"(extraction: {extract_time/60:.1f} | SVM: {svm_time/60:.1f})")
print("=" * 55 + "\n")
print("  Per-Class Report:")
print(classification_report(y_test, y_pred_app1, target_names=CLASS_NAMES))

# Confusion matrix — Approach 1
plt.figure(figsize=(10, 8))
sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, linewidths=0.5)
plt.title("Confusion Matrix — Approach 1 (EfficientNet-B0 + SVM)", fontsize=14, pad=15)
plt.ylabel("True Label", fontsize=12)
plt.xlabel("Predicted Label", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix_approach1.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: confusion_matrix_approach1.png\n")


# ================================================================
#  REQUIREMENT 3 — IMPLEMENTATION (continued)
# ================================================================
print("=" * 60)
print("  REQUIREMENT 3 — Approach 2: End-to-End EfficientNet-B0")
print("=" * 60)

# Build model — same EfficientNet-B0 but with a trainable classifier head
model_e2e = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
)

# Replace classifier: Dropout → Linear(1280 → 10 classes)
in_features = model_e2e.classifier[1].in_features   # 1280
model_e2e.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),   # regularization — reduces overfitting
    nn.Linear(in_features, 10)          # 10 output logits, one per CIFAR-10 class
)

model_e2e = model_e2e.to(device)
print(f"  EfficientNet-B0 loaded (unfrozen) | Custom head: Linear({in_features} → 10)\n")

# Loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model_e2e.parameters(),
                       lr=LEARNING_RATE,
                       weight_decay=WEIGHT_DECAY)

# ReduceLROnPlateau: reduces LR by factor=0.5 if val_loss doesn't improve for 3 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=3, factor=0.5, verbose=True
)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Runs one full pass over the training set.
    Returns average loss and accuracy for this epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()               # clear gradients from previous step
        outputs = model(images)             # forward pass
        loss    = criterion(outputs, labels) # compute cross-entropy loss
        loss.backward()                     # backpropagation
        optimizer.step()                    # update weights

        total_loss += loss.item() * images.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


def evaluate_model(model, loader, criterion, device):
    """
    Evaluates the model on a given DataLoader (validation or test).
    Returns average loss and accuracy.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs     = model(images)
            loss        = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


# ── Training Loop ─────────────────────────────────────────────────────
print(f"  Starting training — {NUM_EPOCHS} epochs | LR: {LEARNING_RATE} | Batch: {BATCH_SIZE}\n")

history = {
    "train_loss": [], "val_loss": [],
    "train_acc":  [], "val_acc":  []
}

best_val_acc    = 0.0
best_model_path = "best_efficientnet_e2e.pth"
start_e2e       = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    ep_start = time.time()

    tr_loss, tr_acc = train_one_epoch(model_e2e, train_loader, optimizer, criterion, device)
    vl_loss, vl_acc = evaluate_model(model_e2e, val_loader, criterion, device)

    # Step the scheduler based on validation loss
    scheduler.step(vl_loss)

    # Save checkpoint if this epoch produced the best validation accuracy
    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model_e2e.state_dict(), best_model_path)
        tag = "  [saved]"
    else:
        tag = ""

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)

    ep_sec = time.time() - ep_start
    print(f"  Epoch {epoch:02d}/{NUM_EPOCHS} | "
          f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc*100:.2f}% | "
          f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc*100:.2f}% | "
          f"{ep_sec:.1f}s{tag}")

total_e2e_time = time.time() - start_e2e
print(f"\n  Training complete in {total_e2e_time/60:.1f} min")
print(f"  Best validation accuracy: {best_val_acc*100:.2f}%\n")

# ── Approach 2 Evaluation (best checkpoint) ───────────────────────────
# Load the best saved weights before evaluating on the test set
model_e2e.load_state_dict(torch.load(best_model_path, map_location=device))
model_e2e.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds  = model_e2e(images).argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(labels.numpy())

y_pred_app2 = np.array(all_preds)
y_true_app2 = np.array(all_true)

acc2  = accuracy_score(y_true_app2, y_pred_app2)
prec2 = precision_score(y_true_app2, y_pred_app2, average="weighted", zero_division=0)
rec2  = recall_score(y_true_app2, y_pred_app2, average="weighted", zero_division=0)
f1_2  = f1_score(y_true_app2, y_pred_app2, average="weighted", zero_division=0)
cm2   = confusion_matrix(y_true_app2, y_pred_app2)
time2 = total_e2e_time / 60

print("=" * 55)
print("  RESULTS — Approach 2 (End-to-End EfficientNet-B0)")
print("=" * 55)
print(f"  Accuracy  : {acc2 * 100:.2f}%")
print(f"  Precision : {prec2 * 100:.2f}%")
print(f"  Recall    : {rec2 * 100:.2f}%")
print(f"  F1-Score  : {f1_2 * 100:.2f}%")
print(f"  Total Time: {time2:.1f} min")
print("=" * 55 + "\n")
print("  Per-Class Report:")
print(classification_report(y_true_app2, y_pred_app2, target_names=CLASS_NAMES))

# Learning curves
epochs_range = range(1, NUM_EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs_range, history["train_loss"], marker="o", label="Train Loss")
axes[0].plot(epochs_range, history["val_loss"],   marker="o", label="Val Loss")
axes[0].set_title("Loss Curve — Approach 2 (End-to-End)", fontsize=13)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(True, alpha=0.4)

axes[1].plot(epochs_range, [a * 100 for a in history["train_acc"]], marker="o", label="Train Acc")
axes[1].plot(epochs_range, [a * 100 for a in history["val_acc"]],   marker="o", label="Val Acc")
axes[1].set_title("Accuracy Curve — Approach 2 (End-to-End)", fontsize=13)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
axes[1].legend(); axes[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("learning_curves_approach2.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: learning_curves_approach2.png\n")

# Confusion matrix — Approach 2
plt.figure(figsize=(10, 8))
sns.heatmap(cm2, annot=True, fmt="d", cmap="Oranges",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, linewidths=0.5)
plt.title("Confusion Matrix — Approach 2 (End-to-End EfficientNet-B0)", fontsize=14, pad=15)
plt.ylabel("True Label", fontsize=12)
plt.xlabel("Predicted Label", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("confusion_matrix_approach2.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: confusion_matrix_approach2.png\n")


# ================================================================
#  REQUIREMENT 4 — COMPARATIVE ANALYSIS & INSIGHTS
# ================================================================
print("=" * 60)
print("  REQUIREMENT 4 — Comparative Analysis")
print("=" * 60)

METRICS       = ["Accuracy", "Precision", "Recall", "F1-Score"]
values_app1   = [acc1, prec1, rec1, f1_1]
values_app2   = [acc2, prec2, rec2, f1_2]
COLOR_APP1    = "#4C72B0"   # blue  — Approach 1
COLOR_APP2    = "#DD8452"   # orange — Approach 2

# ── Plot 1: Metrics Bar Chart ─────────────────────────────────────────
x     = np.arange(len(METRICS))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
bars1 = ax.bar(x - width / 2, [v * 100 for v in values_app1],
               width, label="Approach 1 (EfficientNet + SVM)",
               color=COLOR_APP1, alpha=0.88, edgecolor="white")
bars2 = ax.bar(x + width / 2, [v * 100 for v in values_app2],
               width, label="Approach 2 (End-to-End EfficientNet)",
               color=COLOR_APP2, alpha=0.88, edgecolor="white")

# Annotate bar heights
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)

ax.set_xlabel("Metric", fontsize=12)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Approach 1 vs Approach 2 — Performance Comparison\n"
             "CIFAR-10 | EfficientNet-B0", fontsize=13, pad=12)
ax.set_xticks(x)
ax.set_xticklabels(METRICS, fontsize=12)
ax.set_ylim(0, 112)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("comparison_metrics_bar.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: comparison_metrics_bar.png")

# ── Plot 2: Training Time Comparison ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
labels_time = ["Approach 1\n(EfficientNet + SVM)", "Approach 2\n(End-to-End)"]
times       = [time1, time2]
colors_time = [COLOR_APP1, COLOR_APP2]

bars = ax.bar(labels_time, times, color=colors_time, alpha=0.88,
              edgecolor="white", width=0.45)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{t:.1f} min", ha="center", va="bottom", fontsize=11)

ax.set_ylabel("Training Time (minutes)", fontsize=12)
ax.set_title("Training Time Comparison\nCIFAR-10 | EfficientNet-B0", fontsize=13, pad=12)
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("comparison_training_time.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: comparison_training_time.png")

# ── Plot 3: Confusion Matrices Side by Side ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.4, ax=axes[0])
axes[0].set_title(f"Approach 1 — EfficientNet + SVM\nAccuracy: {acc1*100:.2f}%",
                  fontsize=13, pad=10)
axes[0].set_ylabel("True Label", fontsize=11)
axes[0].set_xlabel("Predicted Label", fontsize=11)
axes[0].tick_params(axis="x", rotation=45)

sns.heatmap(cm2, annot=True, fmt="d", cmap="Oranges",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.4, ax=axes[1])
axes[1].set_title(f"Approach 2 — End-to-End EfficientNet\nAccuracy: {acc2*100:.2f}%",
                  fontsize=13, pad=10)
axes[1].set_ylabel("True Label", fontsize=11)
axes[1].set_xlabel("Predicted Label", fontsize=11)
axes[1].tick_params(axis="x", rotation=45)

plt.suptitle("Confusion Matrices — CIFAR-10 | EfficientNet-B0",
             fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("comparison_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: comparison_confusion_matrices.png\n")

# ── Summary Table ─────────────────────────────────────────────────────
print("=" * 62)
print(f"  {'Metric':<18} {'Approach 1 (SVM)':>20} {'Approach 2 (E2E)':>20}")
print("=" * 62)
for m, v1, v2 in zip(METRICS, values_app1, values_app2):
    print(f"  {m:<18} {v1*100:>19.2f}% {v2*100:>19.2f}%")
print(f"  {'Train Time':<18} {time1:>19.1f}m {time2:>19.1f}m")
print("=" * 62)
print("\n  All outputs saved:")
print("    confusion_matrix_approach1.png")
print("    confusion_matrix_approach2.png")
print("    learning_curves_approach2.png")
print("    comparison_metrics_bar.png")
print("    comparison_training_time.png")
print("    comparison_confusion_matrices.png")