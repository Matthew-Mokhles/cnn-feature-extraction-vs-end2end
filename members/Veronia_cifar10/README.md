# CIFAR-10 Image Classification
### EfficientNet-B0 + SVM vs. End-to-End Fine-Tuning

> **Course:** CAI3105 / CS460 — Deep Learning | **University:** South Valley University | **Week:** 12th Week Project

---

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Implementation](#implementation)
- [Hyperparameter Comparison](#hyperparameter-comparison)
- [Output Files](#output-files)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Comparative Analysis](#comparative-analysis)
- [Citation](#citation)

---

## Project Overview

This project implements and compares two distinct approaches to image classification on the **CIFAR-10** dataset using **EfficientNet-B0** as the backbone architecture:

- **Approach 1:** EfficientNet-B0 as a frozen feature extractor → SVM classifier
- **Approach 2:** End-to-end fine-tuning of EfficientNet-B0

Both approaches are evaluated on Accuracy, Precision, Recall, F1-Score, and Training Time.

---

## Project Structure

```
┌──────────────────────────────────────────────────────────┐
│ REQUIREMENT 1 — Dataset Selection & Preprocessing        │
│ REQUIREMENT 2 — DL Model Selection & Hyperparameters     │
│ REQUIREMENT 3 — Implementation                           │
│   ├── Approach 1: EfficientNet (Feature Extractor) + SVM │
│   └── Approach 2: End-to-End EfficientNet Fine-Tuning    │
│ REQUIREMENT 4 — Comparative Analysis & Visualizations    │
└──────────────────────────────────────────────────────────┘
```

```
project/
├── solution.py                        # Main script (all 4 requirements)
├── best_efficientnet_e2e.pth          # Best model checkpoint (generated)
├── confusion_matrix_approach1.png     # (generated)
├── confusion_matrix_approach2.png     # (generated)
├── learning_curves_approach2.png      # (generated)
├── comparison_metrics_bar.png         # (generated)
├── comparison_training_time.png       # (generated)
├── comparison_confusion_matrices.png  # (generated)
└── data/                              # CIFAR-10 auto-downloaded here
```

---

## Dataset

### CIFAR-10

| Property | Value |
|---|---|
| Source | `torchvision.datasets` (auto-downloaded) |
| Total Images | 60,000 |
| Training Samples | 45,000 (90% of train split) |
| Validation Samples | 5,000 (10% of train split) |
| Test Samples | 10,000 |
| Image Resolution | 32×32 → resized to 224×224 |
| Channels | RGB (3 channels) |
| Classes | 10 |

**Classes:** `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

### Preprocessing & Data Augmentation

**Training transform** (augmentation applied):
| Transform | Justification |
|---|---|
| `Resize(224×224)` | Required input size for EfficientNet-B0 |
| `RandomHorizontalFlip(p=0.5)` | CIFAR-10 objects are symmetric — valid augmentation |
| `RandomRotation(±15°)` | Orientation invariance for real-world images |
| `ColorJitter(brightness=0.2, contrast=0.2)` | Robustness to varying lighting conditions |
| `Normalize(ImageNet mean/std)` | Matches the distribution EfficientNet was trained on |

**Validation / Test transform** (no augmentation — prevents data leakage):
- `Resize(224×224)` → `ToTensor` → `Normalize`

---

## Model Architecture

### EfficientNet-B0

EfficientNet-B0 (Tan & Le, 2019) is selected as the backbone for both approaches. It uses compound scaling across depth, width, and resolution, achieving high accuracy with far fewer parameters than VGG or ResNet.

| Property | Value |
|---|---|
| Parameters | ~5.3 million |
| ImageNet Top-1 Accuracy | 77.1% |
| Feature Output Dimension | 1,280 |
| Core Block | MBConv (Mobile Inverted Bottleneck + Squeeze-Excitation) |
| Pretrained Weights | ImageNet (`IMAGENET1K_V1`) |

> **Reference:** Tan, M., & Le, Q. V. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)

---

## Implementation

### Approach 1 — EfficientNet as Feature Extractor + SVM

**Pipeline:**
1. Load EfficientNet-B0 with pretrained ImageNet weights
2. Replace the classifier with `nn.Identity()` → outputs 1,280-dim feature vectors
3. Freeze all parameters (no gradient updates)
4. Extract feature matrices: `X_train (N, 1280)` and `X_test (N, 1280)`
5. Apply `StandardScaler` — fit on train only (no data leakage)
6. Train a linear SVM on the scaled features
7. Evaluate: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

**SVM Hyperparameters:**

| Hyperparameter | Value |
|---|---|
| Kernel | Linear |
| C (Regularization) | 1.0 |
| Max Iterations | 2,000 |
| Random State | 42 |

---

### Approach 2 — End-to-End Fine-Tuning

**Pipeline:**
1. Load EfficientNet-B0 with pretrained ImageNet weights
2. Replace classifier with `Dropout(0.3)` → `Linear(1280 → 10)`
3. All layers **unfrozen** — entire model is fine-tuned
4. Train with `CrossEntropyLoss`, `Adam` optimizer, `ReduceLROnPlateau` scheduler
5. Save best checkpoint (highest validation accuracy)
6. Evaluate best checkpoint on the test set
7. Plot learning curves (loss + accuracy over epochs)

**Training Hyperparameters:**

| Hyperparameter | Value |
|---|---|
| Epochs | 10|
| Batch Size | 64 |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| Dropout | 0.3 |
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |

---

## Hyperparameter Comparison

| Hyperparameter | Approach 1 (SVM) | Approach 2 (E2E) |
|---|---|---|
| Pretrained Weights | ImageNet | ImageNet |
| Input Size | 224×224 | 224×224 |
| Batch Size | 64 | 64 |
| EfficientNet Frozen | Yes (all layers) | No (fine-tuned) |
| Optimizer | N/A | Adam |
| Learning Rate | N/A | 1e-4 |
| Weight Decay | N/A | 1e-4 |
| Epochs | N/A | 15 |
| Dropout | N/A | 0.3 |
| SVM Kernel | Linear | N/A |
| SVM C | 1.0 | N/A |

---

## Output Files

| File | Description |
|---|---|
| `confusion_matrix_approach1.png` | Confusion matrix — EfficientNet + SVM |
| `confusion_matrix_approach2.png` | Confusion matrix — End-to-End EfficientNet |
| `learning_curves_approach2.png` | Loss & accuracy curves over 15 epochs (Approach 2) |
| `comparison_metrics_bar.png` | Side-by-side bar chart of Accuracy, Precision, Recall, F1 |
| `comparison_training_time.png` | Training time comparison bar chart |
| `comparison_confusion_matrices.png` | Both confusion matrices plotted side by side |
| `best_efficientnet_e2e.pth` | Best model checkpoint (highest validation accuracy) |

---

## Requirements

```bash
pip install torch torchvision scikit-learn numpy matplotlib seaborn
```

- **Python:** 3.8 or higher
- **GPU (recommended):** CUDA-enabled GPU with ≥4 GB VRAM for Approach 2
- **CPU:** Supported but significantly slower for end-to-end training

---

## How to Run

```bash
python solution.py
```

The script will automatically:
1. Detect available GPU/CPU and configure accordingly
2. Download CIFAR-10 to `./data/` on first run
3. Run **Approach 1** — feature extraction + SVM training + evaluation
4. Run **Approach 2** — end-to-end fine-tuning + evaluation
5. Print all results to console and save all output figures

---

## Comparative Analysis

### Approach Comparison

| Aspect | Approach 1 (SVM) | Approach 2 (E2E) |
|---|---|---|
| Training Time | Faster (extraction only) | Slower (full backprop, 15 epochs) |
| Flexibility | Fixed features, no adaptation | Full adaptation to CIFAR-10 |
| Expected Accuracy | Good (ImageNet features transfer) | Higher (task-specific tuning) |
| GPU Memory Usage | Low (inference only) | High (gradient storage) |
| Best For | Resource-constrained environments | High-performance environments |

### Design Decisions

- **`StandardScaler` fit on train only** — prevents data leakage into the SVM evaluation.
- **`ReduceLROnPlateau` scheduler** — halves LR when validation loss stagnates for 3 epochs, improving convergence stability.
- **Best checkpoint evaluation** — the best validation checkpoint (not the final epoch) is used for test set evaluation to avoid overfitting bias.
- **Fixed random seed (42)** — used for the train/validation split to ensure reproducibility.

---

## Citation

```
Tan, M., & Le, Q. V. (2019).
EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
ICML 2019. arXiv:1905.11946
```

---

*CAI3105 / CS460 — Deep Learning — South Valley University — 12th Week Project*
