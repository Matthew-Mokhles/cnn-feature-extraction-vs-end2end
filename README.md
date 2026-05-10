# CNN Feature Extraction vs End-to-End Fine-Tuning

> A comparative study across four image-classification datasets using four different CNN backbones.

**Course:** CAI3105 / CS460 — Deep Learning
**Institution:** South Valley University — College of Computing & Information Technology
**Lecturer:** Prof. Nashwa El-Bendary
**Deadline:** May 7th, 2026 · **Marks:** 20

---

## Table of Contents

- [The Research Question](#the-research-question)
- [Team & Datasets](#team--datasets)
- [Repository Layout](#repository-layout)
- [The Common 6-Step Pipeline](#the-common-6-step-pipeline)
- [Quick Start](#quick-start)
- [Member Modules — Detailed](#member-modules--detailed)
  - [Matthew — Chest X-Ray + ResNet50](#matthew--chest-x-ray--resnet50)
  - [Mark — Chest X-Ray + MobileNetV2](#mark--chest-x-ray--mobilenetv2)
  - [Mario — PlantVillage + EfficientNet-B0](#mario--plantvillage--efficientnet-b0)
  - [Veronia — CIFAR-10 + EfficientNet-B0](#veronia--cifar-10--efficientnet-b0)
- [Cross-Member Results Snapshot](#cross-member-results-snapshot)
- [Evaluation Metrics](#evaluation-metrics)
- [Tech Stack & Requirements](#tech-stack--requirements)
- [Git & Collaboration Conventions](#git--collaboration-conventions)
- [Common Pitfalls & Fixes](#common-pitfalls--fixes)
- [References](#references)

---

## The Research Question

> **Given a pretrained CNN, is it better to use it as a frozen feature extractor and train a classical ML head on top, or to fine-tune the whole network end-to-end?**

To answer this rigorously, the same question is studied **four times** — on four very different datasets and with four different backbones — so we can see whether the answer is task-dependent.

| | Approach 1 — Feature Extraction | Approach 2 — End-to-End |
|---|---|---|
| **Backbone** | Frozen ImageNet-pretrained CNN | Same CNN, fine-tuned |
| **Top of network** | Removed → classical ML head (SVM / LR / MLP) | Custom Dense head, trained jointly |
| **What learns** | Only the ML classifier | The CNN body **plus** the new head |
| **Pros** | Fast, small footprint, low memory | More accurate, adapts features to the task |
| **Cons** | Stuck with generic ImageNet features | Slow, overfits more easily, GPU-hungry |

Both approaches are evaluated on the **same held-out test set** with identical metrics, so the comparison is fair.

---

## Team & Datasets

Each team member implements **both** approaches on their own dataset, in a fully self-contained workspace.

| Alias | Folder | Dataset | Backbone | Approach 1 (Feature Extr.) | Approach 2 (End-to-End) | Status |
|---|---|---|---|---|---|---|
| **Matthew** | [`members/Matthew_xray/`](members/Matthew_xray/) | Chest X-Ray (binary) | ResNet50 | SVM · LR · MLP on GAP+GMP (4096-d) | Two-phase fine-tune + SWA + 8-view TTA | **READY** |
| **Mark** | [`members/mark_xray/`](members/mark_xray/) | Chest X-Ray (binary) | MobileNetV2 | Linear SVM on 1280-d features | Two-phase fine-tune (last 20 layers) | **READY** |
| **Mario** | [`members/Mario_plantvillage/`](members/Mario_plantvillage/) | PlantVillage (38 classes) | EfficientNet-B0 | RBF-SVM + PCA(256) on 1280-d features | Two-phase fine-tune (from layer 100) | **READY** |
| **Veronia** | [`members/Veronia_cifar10/`](members/Veronia_cifar10/) | CIFAR-10 (10 classes) | EfficientNet-B0 (PyTorch) | Linear SVM on 1280-d features | Full fine-tune w/ ReduceLROnPlateau | **READY** |

> Each member's folder is **fully standalone** — its own `config.py`, `utils/`, step scripts, `models/`, and `results/`. No cross-imports between members. This means you can run any one pipeline without touching the others.

---

## Repository Layout

```
cnn-feature-extraction-vs-end2end/
│
├── README.md                       ← This file
├── main.py                         ← Single dispatcher: --member / --step
├── .gitignore                      ← Excludes data/, models/, *.h5, *.npy, *.joblib
│
└── members/                        ← Each person's standalone workspace
    │
    ├── Matthew_xray/               ← ResNet50 + Chest X-Ray
    │   ├── README.md               ← Detailed module README
    │   ├── config.py               ← Hyperparameters, paths, flags
    │   ├── run.py                  ← Local entry point: --step 1..7 / all
    │   ├── dataset_analysis.py     ← Step 1
    │   ├── preprocessing.py        ← Step 2
    │   ├── feature_extraction.py   ← Step 3
    │   ├── ml_classifier.py        ← Step 4
    │   ├── endtoend_dl.py          ← Step 5
    │   ├── evaluation.py           ← Step 6
    │   ├── data_pipeline.py        ← Optional faster tf.data pipeline
    │   ├── utils/{metrics,visualization}.py
    │   ├── tools/split_train_val.py← Enlarge val split from train
    │   ├── requirements.txt
    │   ├── data/chest_xray/        ← Local images (gitignored)
    │   ├── models/                 ← Saved weights (gitignored)
    │   └── results/                ← Plots committed after a run
    │
    ├── mark_xray/                  ← MobileNetV2 + Chest X-Ray
    │   ├── README.md
    │   ├── config.py, run.py, step files, utils/, models/, results/
    │   └── ChestXRay_WithImprovement.ipynb   ← Full Colab notebook
    │
    ├── Mario_plantvillage/         ← EfficientNet-B0 + PlantVillage (38 cls)
    │   ├── README.md
    │   ├── plantvillage_efficientnet.py      ← Single-script implementation
    │   ├── config.py, run.py, utils/, results/
    │
    └── Veronia_cifar10/            ← EfficientNet-B0 + CIFAR-10 (PyTorch)
        ├── README.md
        ├── cifar_effecientNet.py             ← Single-script implementation
        ├── config.py, run.py, Results/
```

---

## The Common 6-Step Pipeline

Every member's workspace follows the same numbered structure. This guarantees an apples-to-apples comparison.

| Step | File | Purpose |
|---|---|---|
| **1** | `dataset_analysis.py` | EDA — class counts, sample grid, image-size stats, intensity histograms |
| **2** | `preprocessing.py` | Build train/val/test generators, visualise augmentations |
| **3** | `feature_extraction.py` | Frozen CNN → cached `.npy` feature vectors |
| **4** | `ml_classifier.py` | Classical ML on extracted features (SVM, LR, etc.) |
| **5** | `endtoend_dl.py` | End-to-end fine-tuning of the CNN |
| **6** | `evaluation.py` | Comparative plots, confusion matrices, training-time chart, summary |

The pipeline is **idempotent and resumable** — if you only tweak the SVM grid, you only re-run Step 4 + Step 6.

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd cnn-feature-extraction-vs-end2end

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate          # macOS / Linux

pip install -r members/Matthew_xray/requirements.txt
```

> Each member has their own `requirements.txt`. Install whichever you plan to run; they overlap heavily but include backbone-specific pins (e.g. PyTorch for Veronia, TF 2.10.1 for Matthew on Windows).

### 2. Download your dataset

| Member | Dataset | Source | Place at |
|---|---|---|---|
| Matthew / Mark | Chest X-Ray Pneumonia | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | `members/<name>/data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/` |
| Mario | PlantVillage (color subset) | [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) | `members/Mario_plantvillage/data/plant_village/` |
| Veronia | CIFAR-10 | Auto-downloads via `torchvision.datasets` | — |

### 3. Run a member's pipeline

```bash
# Recommended — top-level dispatcher (only Matthew is wired into main.py at the moment)
python main.py --member matthew --step all
python main.py --member matthew --step 1     # EDA only
python main.py --member matthew --step 5     # Just retrain end-to-end

# Or run a member's run.py directly
python members/Matthew_xray/run.py --step 4
python members/mark_xray/run.py
python members/Mario_plantvillage/run.py
python members/Veronia_cifar10/run.py
```

---

## Member Modules — Detailed

### Matthew — Chest X-Ray + ResNet50

**Task:** Binary classification (`NORMAL` vs `PNEUMONIA`) on 6,638 anterior-posterior chest radiographs.

**Splits:** Train 5,216 (74.3% PNEUMONIA) · Val 798 (73.8%) · Test 624 (62.5%)
The **distribution shift** between train/val (~74% pneumonia) and test (~63%) is the central modelling challenge.

#### Approach 1 — Feature Extraction
- **ResNet50** (ImageNet, frozen) → **GAP + GMP** concatenated → **4096-d** vector per image
- Three classifiers wrapped in `StandardScaler → L2-row-normalise → Estimator`:
  - **SVM** — linear kernel, `GridSearchCV` over `C ∈ {0.01, 0.1, 1.0}`, balanced class weights
  - **Logistic Regression** — same grid
  - **MLP** — 2 hidden layers (256, 64), Adam, early stopping
- CV scoring is `f1_macro` so a degenerate "always-PNEUMONIA" model is heavily penalised

#### Approach 2 — End-to-End
Two-phase fine-tuning with a deep regularised head:

```
Image (224×224×3)
  → ResNet50 base
  → GAP + GMP  (4096-d)
  → Dense(512) + BN + ReLU + Dropout(0.4) + L2(1e-4)
  → Dense(128) + BN + ReLU + Dropout(0.3) + L2(1e-4)
  → Dense(1, sigmoid) → P(PNEUMONIA)
```

| Phase | What | Epochs | LR | Layers Trained |
|---|---|---|---|---|
| **1 — Head warm-up** | Base frozen | 10 | Adam(1e-4) + ReduceLROnPlateau | Head only |
| **2 — Fine-tune** | Last 15 base layers unfrozen | 15 | Adam(1e-5) + cosine decay → 5e-7 | 24 / 184 layers |

**Advanced techniques** (all enabled by default):
- **Label smoothing** (0.1) — improves calibration
- **L2 weight decay** (1e-4) on Dense layers
- **BatchNorm frozen in Phase 2** — protects running statistics from small batches
- **Stochastic Weight Averaging (SWA)** — averages weights from the last 40% of Phase 2
- **8-view Test-Time Augmentation** — original + H-flip + V-flip + 90° + 4 corner crops
- **Dual threshold tuning** on validation: F1-optimal **and** Youden's J (more robust under shift)
- **RandomErasing / cutout** (30% probability) — bridges the train↔test domain gap

**Headline test-set results** (from `results/comparison_summary.txt`):

| Model | Accuracy | F1 (weighted) | ROC AUC | PR AUC | Train Time |
|---|---|---|---|---|---|
| SVM (ResNet50 feats) | 0.8045 | 0.7844 | 0.9533 | 0.9699 | 1,705 s |
| LR  (ResNet50 feats) | 0.8237 | 0.8110 | 0.9520 | 0.9678 |     16 s |
| MLP (ResNet50 feats) | 0.7837 | 0.7578 | 0.9525 | 0.9678 |     16 s |
| **End-to-End ResNet50** | **0.9103** | **0.9093** | **0.9708** | **0.9785** | 5,923 s |

**Conclusion:** End-to-end wins on every quality metric; LR on extracted features is the cheapest competitive baseline. See the [module README](members/Matthew_xray/README.md) for full details.

---

### Mark — Chest X-Ray + MobileNetV2

**Task:** Same binary chest-X-ray pneumonia task, but with **MobileNetV2** (~3.4M parameters) — a lightweight backbone designed for mobile/embedded inference.

#### Approach 1 — Feature Extractor + SVM
```
Image → MobileNetV2 (frozen) → 1280 features → StandardScaler → SVM (linear, C=1.0)
```

#### Approach 2 — End-to-End (two variants)
```
Image → MobileNetV2 (last 20 layers unfrozen)
      → GAP → BN → Dropout(0.4)
      → Dense(128, ReLU) → Dropout(0.3)
      → Dense(1, sigmoid)
```

The most interesting story in Mark's module is a **negative result that turns into a positive one**:

| | Approach 2A | Approach 2B |
|---|---|---|
| **Validation set** | Original Kaggle val (16 images) | `validation_split=0.2` from train (1,043 images) |
| **Class weights** | None | Balanced |
| **Fine-tune LR** | 1e-4 | 5e-5 |
| **Result** | Collapsed → predicts PNEUMONIA for everything (NORMAL recall ≈ 0%) | Fully recovered |

> **Key finding:** With only 16 validation images, a single wrong prediction is a 6.25% accuracy swing — early-stopping signal is unreliable, and the model collapses to the majority class. Increasing the val set to 1,043 images and adding balanced class weights raised test accuracy by **+17.31 points**.

**Headline test-set results:**

| Metric | Approach 1 (SVM) | Approach 2A (val=16) | **Approach 2B (val=1043)** |
|---|---|---|---|
| Accuracy | 85.10% | 70.51% | **87.82%** |
| Precision | 0.8322 | 0.6801 | **0.8520** |
| Recall | 0.9538 | 0.9974 | 0.9744 |
| F1-Score | 0.8889 | 0.8087 | **0.9091** |
| AUC | 0.9415 | 0.9249 | **0.9644** |
| Train Time | **8 s** | 1,805 s | 3,190 s |

See the [module README](members/mark_xray/README.md) for the full ablation.

---

### Mario — PlantVillage + EfficientNet-B0

**Task:** **38-class** plant-disease classification on the PlantVillage `color` subset (~55,448 images of 14 crop species, healthy + diseased).

**Splits:** Train 70% (~38,814) · Val 15% (~8,317) · Test 15% (~8,317), seed 42.

#### Approach 1 — End-to-End DL

Two-phase fine-tuning of EfficientNet-B0 (~5.3M parameters):

| Phase | What | Epochs | LR | Notes |
|---|---|---|---|---|
| **1 — Warm-up** | EfficientNet base frozen | 10 | 1e-4 | Train new head only |
| **2 — Fine-tune** | Layers 100+ unfrozen (~136 trainable) | up to 20 | 1e-5 | Early stop @ patience 5, ReduceLROnPlateau |

Head: `BN → Dropout(0.5) → Dense(38, softmax)`

#### Approach 2 — Feature Extractor + RBF-SVM

```
Image → EfficientNet-B0 (frozen) → 1280-D features
      → PCA (1280 → 256, retains >95% variance)
      → StandardScaler
      → RBF-SVM (C=10, gamma="scale")
```

PCA before SVM dramatically speeds up the kernel computation while preserving virtually all of the discriminative signal. RBF (not linear) is chosen because EfficientNet's feature space is highly non-linear in 256 dimensions.

**Headline test-set results** (from `results/results.json`):

| Metric | Approach 1 (End-to-End) | Approach 2 (Features + SVM) |
|---|---|---|
| Accuracy | **99.78%** | 97.74% |
| Precision | **99.78%** | 97.83% |
| Recall | **99.78%** | 97.74% |
| F1-Score | **99.78%** | 97.73% |
| Train Time | ~13.5 h | ~8 min |
| Inference Time | 772 s | 1,074 s |

> On a clean, large, low-noise dataset, end-to-end pulls ahead by ~2 absolute points — but the feature-extractor pipeline trains **~100× faster** and is competitive enough for many practical deployments.

See the [module README](members/Mario_plantvillage/README.md) for the full hyperparameter table and per-class breakdown.

---

### Veronia — CIFAR-10 + EfficientNet-B0 (PyTorch)

**Task:** Classic 10-class object recognition on CIFAR-10 (60,000 images, 32×32 → upscaled to 224×224).

**Splits:** 45,000 train · 5,000 val · 10,000 test (CIFAR-10's standard split, then a 90/10 split of train into train/val with seed 42).

This is the only **PyTorch** module in the repo (the others use TensorFlow/Keras).

#### Approach 1 — Feature Extractor + Linear SVM

```
Image → EfficientNet-B0 (frozen, classifier replaced with nn.Identity)
      → 1280-D features → StandardScaler (fit on train only)
      → LinearSVM (C=1.0, max_iter=2000)
```

#### Approach 2 — End-to-End Fine-Tuning

```
Image → EfficientNet-B0 (all layers unfrozen)
      → Dropout(0.3) → Linear(1280 → 10)
```

| Hyperparameter | Value |
|---|---|
| Epochs | 10–15 |
| Batch size | 64 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Dropout | 0.3 |
| Optimizer | Adam |
| Scheduler | `ReduceLROnPlateau` (patience=3, factor=0.5) |
| Loss | CrossEntropyLoss |

**Augmentations** (training only, none at val/test):
`Resize(224) → RandomHorizontalFlip(0.5) → RandomRotation(±15°) → ColorJitter(0.2,0.2) → Normalize(ImageNet stats)`

**Design choices:**
- `StandardScaler` fit on train only — prevents leakage into the SVM evaluation
- `ReduceLROnPlateau` for stability when val loss plateaus
- **Best-checkpoint** evaluation (highest validation accuracy), not the final epoch
- Fixed seed (42) for reproducible train/val splits

See the [module README](members/Veronia_cifar10/README.md) for the full configuration.

---

## Cross-Member Results Snapshot

A high-level view of how the two paradigms compare on each dataset (best numbers per member).

| Member | Dataset | Backbone | Best Feat-Extr. Acc. | Best End-to-End Acc. | Winner | Gap |
|---|---|---|---|---|---|---|
| Matthew | Chest X-Ray (binary) | ResNet50 | 82.4% (LR) | **91.0%** | End-to-End | +8.6 |
| Mark | Chest X-Ray (binary) | MobileNetV2 | 85.1% (SVM) | **87.8%** (2B) | End-to-End | +2.7 |
| Mario | PlantVillage (38 cls) | EfficientNet-B0 | 97.7% (RBF-SVM) | **99.8%** | End-to-End | +2.0 |
| Veronia | CIFAR-10 (10 cls) | EfficientNet-B0 | (pending) | (pending) | — | — |

**Recurring observations across members:**

1. **End-to-end consistently wins on accuracy**, but the margin shrinks dramatically on cleaner datasets (Mario: +2 points on PlantVillage) and grows on harder ones (Matthew: +8.6 points under distribution shift).
2. **Feature extraction is 10×–100× faster to train** (LR on extracted features: 16 s vs end-to-end: 5,923 s for Matthew).
3. **Validation set quality matters more than model architecture.** Mark's experiment showed that growing val from 16 → 1,043 images recovered +17 accuracy points, far more than any architectural tweak.
4. **GAP+GMP > GAP alone** by ~3–5 points on chest X-rays (Matthew's finding).
5. **Threshold tuning + SWA + TTA** combine to give a final ~1–2 point lift on the harder binary task.

---

## Evaluation Metrics

All members report the standard set:

- **Accuracy**
- **Precision / Recall / F1** (class-weighted for imbalanced datasets, macro for balanced ones)
- **Confusion Matrix** (raw + normalised)
- **Training time** (wall-clock seconds)
- **Training curves** (loss + accuracy per epoch)

Additional metrics where applicable:
- **ROC AUC + PR AUC** (binary tasks — Matthew, Mark)
- **Per-class accuracy** (multi-class tasks — Mario)
- **Calibration plot** + **threshold sweep** (Matthew)

For **Matthew specifically**, the comparison table uses **class-weighted** Precision/Recall/F1 because the class imbalance (~3:1) and distribution shift make plain accuracy misleading. Per-class detail remains in `classification_report` and confusion matrices. ML models are fit on **train + val features** and reported scores are evaluated on the **held-out test set only**.

---

## Tech Stack & Requirements

| Library | Used By | Purpose |
|---|---|---|
| `tensorflow` (2.10.1 on Win, ≥2.11 elsewhere) | Matthew, Mark, Mario | Backbones (ResNet50, MobileNetV2, EfficientNet-B0), training |
| `torch` / `torchvision` | Veronia | EfficientNet-B0 backbone + CIFAR-10 download |
| `scikit-learn` ≥ 1.2 | All | SVM, LR, MLP, GridSearchCV, metrics, PCA, StandardScaler |
| `numpy`, `pandas` | All | Data wrangling |
| `matplotlib`, `seaborn` | All | Plots, confusion-matrix heatmaps |
| `Pillow` | All | Image I/O |
| `tqdm` | TF members | Progress bars |
| `joblib` | TF members | Saving SVM/LR/MLP pipelines |

**Python:** 3.8 – 3.10 (TF 2.10 wheel constraint on Windows).

**GPU notes:**
- Matthew's `requirements.txt` pins **TF 2.10.1 on Windows** because TF 2.11+ dropped native Windows GPU support. CUDA 11.2 + cuDNN 8.1 is recommended.
- Veronia's PyTorch module supports CPU + GPU automatically.
- Long CPU runtimes are normal for the end-to-end steps; expect hours.

---

## Git & Collaboration Conventions

1. **Never push** `data/`, `models/`, `*.npy`, `*.h5`, `*.joblib`, `*.pkl` — already in `.gitignore`.
2. **Each member only edits their own** `members/<name>/` folder. The repo is structured so there are no shared imports between members.
3. **Push your `members/<name>/results/` plots** after a successful run. They are the deliverable.
4. **Branch per person:** `git checkout -b your-name` → PR to `main`.
5. Each member's folder is fully standalone — no shared `core/` package.

---

## Common Pitfalls & Fixes

| Problem | Fix |
|---|---|
| `TypeError: unsupported operand type(s) for *: 'CosineDecay' and 'int'` | Known TF 2.10 / Windows bug. Matthew's pipeline works around it by passing the LR via `LearningRateScheduler` callback (returns a plain float) instead of feeding `CosineDecay` to `Adam()`. |
| `Could not load dynamic library 'cudart64_110.dll'` | Harmless if no GPU (TF falls back to CPU). For GPU, install [CUDA 11.2 + cuDNN 8.1](https://www.tensorflow.org/install/pip#windows-native). |
| Tiny default Kaggle val set (16 images) | Use `members/Matthew_xray/tools/split_train_val.py --fraction 0.15 --copy` to carve a stratified val split from train. |
| `MemoryError` during 8-view TTA (Matthew) | Reduce `BATCH_SIZE` in `config.py` or set `USE_TTA = False`. |
| Mario: `FileNotFoundError: Dataset not found` | Use `r"..."` raw strings on Windows for the dataset path in `CFG`. |
| Mario: SVM takes too long | Reduce `pca_components` from 256 to 128. |
| Out of memory (any member) | Reduce `batch_size` in `config.py` from 32 → 16. |
| Validation `val_accuracy` is unstable | Almost always a too-small val set. Aim for **hundreds of images per class** (~10–20% of train), or merge pools and use k-fold. |

### TensorFlow on Windows — quietening logs

```powershell
$env:TF_CPP_MIN_LOG_LEVEL='2'   # PowerShell
```

```bash
export TF_CPP_MIN_LOG_LEVEL=2   # Linux/macOS
```

Matthew's `run.py` sets a quiet default via `setdefault` so you can override if needed.

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition.** *CVPR 2016.* — ResNet50
2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). **MobileNetV2: Inverted Residuals and Linear Bottlenecks.** *CVPR 2018.*
3. Tan, M., & Le, Q. V. (2019). **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.** *ICML 2019.* [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
4. Cortes, C., & Vapnik, V. (1995). **Support-Vector Networks.** *Machine Learning, 20(3), 273–297.*
5. Mohanty, S. P., Hughes, D. P., & Salathe, M. (2016). **Using Deep Learning for Image-Based Plant Disease Detection.** *Frontiers in Plant Science, 7, 1419.*
6. Mooney, P. (2018). **Chest X-Ray Images (Pneumonia).** *Kaggle Dataset.*
7. Krizhevsky, A. (2009). **Learning Multiple Layers of Features from Tiny Images.** *Tech Report, Univ. of Toronto.* — CIFAR-10
8. Izmailov, P. et al. (2018). **Averaging Weights Leads to Wider Optima and Better Generalization.** *UAI 2018.* — SWA
9. Zhong, Z. et al. (2020). **Random Erasing Data Augmentation.** *AAAI 2020.*
10. Youden, W. J. (1950). **Index for rating diagnostic tests.** *Cancer, 3(1), 32–35.* — Youden's J statistic

---

## Course Information

| | |
|---|---|
| **Course** | CAI3105 / CS460 — Deep Learning |
| **Institution** | South Valley University · College of Computing & IT |
| **Lecturer** | Prof. Nashwa El-Bendary |
| **Project** | Week 12 — CNN Feature Extraction vs End-to-End Fine-Tuning |
| **Deadline** | Thursday, May 7th 2026 (11:55 PM) |
| **Platform** | Moodle LMS |
| **Marks** | 20 |

---

*For deep-dive details on any individual experiment, see the per-member READMEs:*
*[Matthew](members/Matthew_xray/README.md) · [Mark](members/mark_xray/README.md) · [Mario](members/Mario_plantvillage/README.md) · [Veronia](members/Veronia_cifar10/README.md)*
