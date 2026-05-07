# Matthew — Chest X-Ray Pneumonia Detection with ResNet50

> **CNN Feature Extraction vs End-to-End Fine-Tuning**  
> Binary classification: **NORMAL** vs **PNEUMONIA** on the [Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Pipeline Steps](#pipeline-steps)
- [Architecture Details](#architecture-details)
- [Approach 1 — Feature Extraction + ML Classifiers](#approach-1--feature-extraction--ml-classifiers)
- [Approach 2 — End-to-End Fine-Tuning](#approach-2--end-to-end-fine-tuning)
- [Key Techniques](#key-techniques)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Configuration Reference](#configuration-reference)
- [Results & Outputs](#results--outputs)
- [Troubleshooting](#troubleshooting)

---

## Overview

This module implements both approaches of the comparative study:

| Approach | Method | Models |
|----------|--------|--------|
| **1 — Feature Extraction** | Freeze ResNet50, extract GAP+GMP embeddings (4096-d), train classical ML heads | SVM, Logistic Regression, MLP |
| **2 — End-to-End** | Two-phase fine-tuning of ResNet50 with a deep regularised head | ResNet50 + Dense(512) + Dense(128) |

Both approaches are evaluated on the same held-out test set with threshold tuning, ROC/PR curves, calibration plots, and confusion matrices.

---

## Project Structure

```
members/Matthew_xray/
├── config.py               # All hyperparameters, paths, and flags
├── run.py                  # Local entry point (--step 1-7 or all)
├── __init__.py
│
├── dataset_analysis.py     # Step 1 — EDA, class distribution, intensity histograms
├── preprocessing.py        # Step 2 — Data generators, augmentation visualisation
├── feature_extraction.py   # Step 3 — GAP+GMP feature extraction from frozen ResNet50
├── ml_classifier.py        # Step 4 — SVM, LR, MLP with GridSearchCV
├── endtoend_dl.py          # Step 5 — Two-phase end-to-end fine-tuning
├── evaluation.py           # Step 6 — Comparative analysis, threshold tuning, plots
│
├── data_pipeline.py        # Optional tf.data pipeline with RandomErasing
│
├── utils/
│   ├── metrics.py          # Compute metrics, threshold search (F1 + Youden's J)
│   └── visualization.py    # Training curves, ROC, PR, calibration, confusion matrix
│
├── tools/
│   └── split_train_val.py  # Utility to enlarge val split from train
│
├── data/
│   └── chest_xray/
│       ├── train/          # 1,341 NORMAL + 3,875 PNEUMONIA
│       ├── val/            #   209 NORMAL +   589 PNEUMONIA
│       └── test/           #   234 NORMAL +   390 PNEUMONIA
│
├── models/                 # Saved models (.h5, .joblib) + extracted features
│   └── features/           # Cached GAP+GMP feature vectors (.npy)
│
├── results/                # All output plots, metrics, and summaries
│
└── requirements.txt        # Python dependencies
```

---

## Dataset

**Chest X-Ray Pneumonia** — 6,638 anterior-posterior chest radiographs.

| Split | NORMAL | PNEUMONIA | Total | PNEUMONIA % |
|-------|--------|-----------|-------|-------------|
| Train | 1,341  | 3,875     | 5,216 | 74.3%       |
| Val   | 209    | 589       | 798   | 73.8%       |
| Test  | 234    | 390       | 624   | 62.5%       |

> **⚠ Distribution shift:** The test set has a different class ratio (62.5% PNEUMONIA) compared to train/val (~74%). This is a key challenge — the pipeline uses balanced class weights, label smoothing, threshold tuning, and SWA to address it.

### Data Setup

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Extract to `members/Matthew_xray/data/chest_xray/`
3. Ensure the folder structure matches: `data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/`

---

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| **1** | `dataset_analysis.py` | EDA — class distribution, sample images, grayscale detection, pixel intensity histograms, resolution scatter |
| **2** | `preprocessing.py` | Data generators, augmentation visualisation |
| **3** | `feature_extraction.py` | Extract GAP+GMP embeddings (4096-d) from frozen ResNet50 |
| **4** | `ml_classifier.py` | Train SVM, LR, MLP on extracted features with GridSearchCV |
| **5** | `endtoend_dl.py` | Two-phase ResNet50 fine-tuning with SWA and 8-view TTA |
| **6** | `evaluation.py` | Comparative analysis, threshold tuning on val, ROC/PR/calibration plots |
| **7** | *(alias for 6)* | Re-run evaluation without retraining |

---

## Architecture Details

### Approach 1 — Feature Extraction + ML Classifiers

```
Input Image (224×224×3)
    │
    ▼
┌─────────────────────────┐
│  ResNet50 (frozen)      │  ImageNet weights, no top
│  175 conv/BN/pool layers│
└──────────┬──────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
   [GAP]       [GMP]        Global Average + Max Pooling
     │           │
     └─────┬─────┘
           ▼
    Concatenate (4096-d)
           │
     ┌─────┼─────────┐
     ▼     ▼         ▼
   [SVM] [LR]      [MLP]    Classical ML classifiers
```

**Pipeline per model:** `StandardScaler → L2 row-normalisation → Estimator`

- **SVM:** GridSearchCV over C ∈ {0.01, 0.1, 1.0}, linear kernel, balanced class weights
- **Logistic Regression:** GridSearchCV over C ∈ {0.01, 0.1, 1.0}, balanced class weights
- **MLP:** 2-layer (256, 64), Adam, early stopping (validation_fraction=0.1)

### Approach 2 — End-to-End Fine-Tuning

```
Input Image (224×224×3)
    │
    ▼
┌─────────────────────────┐
│  ResNet50 base           │
│  Phase 1: fully frozen   │
│  Phase 2: last 15 unfrozen│
└──────────┬──────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
   [GAP]       [GMP]
     │           │
     └─────┬─────┘
           ▼
    Concatenate (4096-d)
           │
           ▼
    Dense(512) + BN + ReLU + Dropout(0.4) + L2(1e-4)
           │
           ▼
    Dense(128) + BN + ReLU + Dropout(0.3) + L2(1e-4)
           │
           ▼
    Dense(1, sigmoid) → P(PNEUMONIA)
```

#### Two-Phase Training Strategy

| Phase | What | Epochs | LR Schedule | Layers Trained |
|-------|------|--------|-------------|----------------|
| **Phase 1** — Head warm-up | Base frozen, train head only | 10 | Adam(1e-4) + ReduceLROnPlateau | Head only (9 layers) |
| **Phase 2** — Base fine-tuning | Last 15 base layers unfrozen | 15 | Adam(1e-5) + Cosine decay → 5e-7 | 24 / 184 layers |

- **BatchNorm frozen** in Phase 2 — prevents small fine-tuning batches from corrupting running statistics
- **SWA** averages weights from epochs 9–15 of Phase 2 for a flatter minimum
- **Checkpoint** monitors `val_loss` (smoother than `val_accuracy` on small val sets)

---

## Key Techniques

### Regularisation
- **Label smoothing** (0.1) — prevents overconfident predictions, improves calibration
- **L2 weight regularisation** (1e-4) — on head Dense layers
- **Graduated dropout** — 0.4 → 0.3 across head layers
- **Balanced class weights** — corrects PNEUMONIA:NORMAL ≈ 3:1 imbalance

### Data Augmentation
- Rotation (±20°), translation (±15%), zoom (±15%), shear (±8%), brightness (80%–120%), horizontal flip
- **RandomErasing** (cutout) — masks random rectangles with per-channel mean fill (30% probability)
- Designed to close the domain gap between train/val and test distributions

### Inference
- **8-view Test-Time Augmentation:** original + H-flip + V-flip + 90° rotation + 4 corner crops (0.9× scale)
- **Dual threshold tuning on validation:**
  - **F1-optimal** — maximises binary F1 for PNEUMONIA
  - **Youden's J** (TPR − FPR) — more conservative, often better under distribution shift

### Stochastic Weight Averaging (SWA)
Averages model weights from the last 40% of Phase 2 training to find a flatter loss-landscape minimum. A pre-SWA snapshot is saved for comparison.

---

## Setup & Installation

### Prerequisites
- Python 3.8–3.10
- (Optional) NVIDIA GPU + CUDA 11.2 for GPU acceleration

### Install Dependencies

```bash
# From the repository root
cd members/Matthew_xray

# Create virtual environment (recommended)
python -m venv ../../.venv
../../.venv/Scripts/activate    # Windows
# source ../../.venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

> **⚠ Windows GPU note:** TensorFlow 2.11+ dropped native Windows GPU support. This project pins TF 2.10.1 on Windows to retain CUDA capability. See `requirements.txt`.

---

## Usage

### Run the full pipeline (Steps 1–6)

```bash
# From members/Matthew_xray/
python run.py --step all

# Or from the repository root:
python main.py --member matthew --step all
```

### Run individual steps

```bash
python run.py --step 1    # Dataset analysis / EDA
python run.py --step 2    # Preprocessing & augmentation vis
python run.py --step 3    # Feature extraction
python run.py --step 4    # ML classifiers (SVM, LR, MLP)
python run.py --step 5    # End-to-end DL training
python run.py --step 6    # Comparative evaluation
python run.py --step 7    # Re-run evaluation only (alias for 6)
```

### Typical workflow

```bash
# First run: execute everything
python run.py --step all

# After tweaking ML params: re-run only Steps 4 + 6
python run.py --step 4
python run.py --step 6

# After tweaking DL config: re-run only Step 5 + 6
python run.py --step 5
python run.py --step 6
```

---

## Configuration Reference

All settings are centralised in [`config.py`](config.py). Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | (224, 224) | Resize target for all images |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS` | 25 | Total epochs (Phase 1 + Phase 2) |
| `LEARNING_RATE` | 1e-4 | Phase 1 learning rate |
| `DROPOUT_RATE` | 0.4 | Head layer 1 dropout (layer 2 uses 0.3) |
| `LABEL_SMOOTHING` | 0.1 | Binary cross-entropy smoothing |
| `PHASE2_EPOCHS` | 15 | Phase 2 epoch count |
| `PHASE2_UNFREEZE` | 15 | Number of base layers to unfreeze |
| `PHASE2_FREEZE_BN` | True | Keep BatchNorm in inference mode during Phase 2 |
| `EARLY_STOPPING_PATIENCE` | 6 | Epochs to wait before early stopping |
| `USE_SWA` | True | Enable Stochastic Weight Averaging |
| `SWA_START_FRAC` | 0.6 | Start SWA from 60% of Phase 2 |
| `USE_TTA` | True | Enable 8-view Test-Time Augmentation |
| `USE_TF_DATA` | True | Use tf.data pipeline (faster, GPU augmentation) |
| `ML_SCORING` | "f1_macro" | GridSearchCV scoring metric |

---

## Results & Outputs

After running the full pipeline, results are saved to `results/`:

### Plots
| File | Description |
|------|-------------|
| `class_distribution.png` | Bar chart of class counts per split |
| `sample_images.png` | Grid of sample training images |
| `intensity_histogram.png` | NORMAL vs PNEUMONIA pixel intensity distributions |
| `resolution_scatter.png` | Native image resolutions before resizing |
| `augmentation_samples.png` | Original vs augmented image pairs |
| `dl_training_curves.png` | Loss and accuracy curves for Phase 1 + 2 |
| `cm_svm.png` / `cm_lr.png` / `cm_mlp.png` | ML classifier confusion matrices |
| `cm_dl.png` / `cm_dl_tuned.png` | DL confusion matrices (default + tuned threshold) |
| `roc_curves.png` | ROC curves for all models |
| `pr_curves.png` | Precision-Recall curves for all models |
| `calibration_dl.png` | Probability calibration plot for the DL model |
| `threshold_sweep_dl.png` | F1 / precision / recall vs threshold |
| `comparison_metrics.png` | Side-by-side metric bar chart |
| `comparison_time.png` | Training time comparison |
| `all_confusion_matrices.png` | Collage of all confusion matrices |

### Data Files
| File | Description |
|------|-------------|
| `dl_results.npy` | DL model metrics dictionary |
| `ml_results.npy` | ML models metrics dictionary |
| `dl_history.npy` | Full training history (loss, accuracy, LR per epoch) |
| `*_val_proba.npy` / `*_test_proba.npy` | Saved probabilities for threshold tuning |
| `ml_best_params.json` | Best GridSearchCV parameters for SVM and LR |
| `comparison_summary.txt` | Text summary of the comparative analysis |

### Saved Models
| File | Description |
|------|-------------|
| `models/resnet50_endtoend.h5` | Best DL model checkpoint (SWA-averaged) |
| `models/resnet50_endtoend_phase1.h5` | Phase 1 safety snapshot |
| `models/resnet50_endtoend_pre_swa.h5` | Pre-SWA snapshot for comparison |
| `models/svm_classifier.joblib` | Best SVM pipeline |
| `models/lr_classifier.joblib` | Best LR pipeline |
| `models/mlp_classifier.joblib` | Best MLP pipeline |
| `models/features/*.npy` | Cached GAP+GMP feature vectors |

---

## Troubleshooting

### `TypeError: unsupported operand type(s) for *: 'CosineDecay' and 'int'`
This is a known TF 2.10 / Windows bug. The pipeline works around it by delivering the LR schedule as a `LearningRateScheduler` callback (returns a plain float) rather than passing `CosineDecay` directly to `Adam()`.

### `Could not load dynamic library 'cudart64_110.dll'`
TensorFlow can't find the CUDA runtime. If you don't have a GPU, this warning is harmless — TF will fall back to CPU. If you have a GPU, install [CUDA 11.2 + cuDNN 8.1](https://www.tensorflow.org/install/pip#windows-native).

### Small validation set warning
The original Kaggle split has a tiny val set (16 images). The pipeline warns about this and includes `tools/split_train_val.py` to carve a larger stratified val split from train. The current val set has 798 images (after running the split tool).

### `MemoryError` during TTA
8-view TTA runs 8 forward passes per batch. If you run out of memory, reduce `BATCH_SIZE` in `config.py` or set `USE_TTA = False`.

---

## Author

**Matthew** — Chest X-Ray (Pneumonia) + ResNet50  
Part of the CNN Feature Extraction vs End-to-End comparative study.
