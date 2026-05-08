# CNN Feature Extraction vs End-to-End Deep Learning
## CAI3105 / CS460 — Deep Learning | 12th Week Project

**College of Computing and Information Technology — South Valley Campus**  
**Lecturer:** Prof. Nashwa El-Bendary  
**Course:** Deep Learning (CAI3105 / CS460)

---

## 📋 Project Overview

This project evaluates the performance of **End-to-End Deep Learning (DL) classification** against **DL-based Feature Learning** on the Chest X-Ray (Pneumonia) dataset.

| | Approach 1 | Approach 2A | Approach 2B |
|---|---|---|---|
| **Method** | MobileNetV2 + SVM (Linear) | E2E MobileNetV2 (Original) | E2E MobileNetV2 (Improved) |
| **Validation Set** | 1,043 images | ❌ 16 images | ✅ 1,043 images |
| **Accuracy** | 85.10% | 70.51% | **87.82%** |
| **AUC** | 0.9415 | 0.9249 | **0.9644** |
| **Train Time** | **8s** | 1,805s | 3,190s |

---

## 🗂️ Repository Structure

```
cnn-feature-extraction-vs-end2end/
│
├── main.py                          ← Single entry point for all members
├── requirements.txt                 ← Python dependencies
├── .gitignore
│
└── members/
    │
    ├── Matthew_xray/                ← Matthew (MobileNetV2 + X-Ray)
    │   ├── config.py                ← Paths, hyperparameters, settings
    │   ├── run.py                   ← Run all steps for this member
    │   ├── dataset_analysis.py      ← Step 1: EDA & class distribution
    │   ├── preprocessing.py         ← Step 2: Augmentation visualization
    │   ├── feature_extraction.py    ← Step 3: MobileNetV2 feature extractor
    │   ├── ml_classifier.py         ← Step 4: SVM Linear classifier
    │   ├── endtoend_dl.py           ← Step 5: End-to-End fine-tuning (2A & 2B)
    │   ├── evaluation.py            ← Step 6: Metrics, plots, comparison
    │   ├── utils/
    │   │   ├── metrics.py           ← Accuracy, Precision, Recall, F1, AUC
    │   │   └── visualization.py     ← Confusion matrix, ROC, bar charts
    │   ├── models/                  ← Saved model files (gitignored)
    │   └── results/                 ← Output plots (committed after run)
    │       ├── req1_sample_images.png
    │       ├── req1_validation_comparison.png
    │       ├── req1_class_distribution.png
    │       ├── req3_cm_approach1.png
    │       ├── req3_cm_approach2a.png
    │       ├── req3_cm_approach2b.png
    │       ├── req3_learning_curves_2a_vs_2b.png
    │       ├── req4_3way_comparison.png
    │       ├── req4_before_after_val.png
    │       ├── req4_3cm_comparison.png
    │       └── req4_roc_all.png
    │
    ├── mark_xray/                   ← Mark (VGG16 + X-Ray)
    │   └── config.py, run.py, step files, utils/, models/, results/
    │
    ├── Mario_plantvillage/          ← Mario (ResNet50 + PlantVillage 38 cls)
    │   └── config.py, run.py, step files, utils/, models/, results/
    │
    └── Veronia_cifar10/             ← Veronia (ResNet50 + CIFAR-10)
        └── config.py, run.py, step files, utils/, models/, results/
```

---

## 📦 Dataset

**Chest X-Ray Images (Pneumonia)**
- **Source:** [Kaggle — paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Samples:** 5,856 images
- **Classes:** NORMAL (1,583) / PNEUMONIA (4,273)
- **Resolution:** Resized to 224 × 224 × 3 (RGB)

---

## 🏗️ Model Architecture

**MobileNetV2** (Sandler et al., CVPR 2018)
- Input → 32 Conv2D → **19 Inverted Residual Bottleneck Blocks** → 1280 Conv2D → GlobalAvgPool → Output
- Pre-trained on ImageNet (1.28M images, 1000 classes)
- Only **3.4M parameters** — lightweight and efficient

### How it is used in each approach:

```
Approach 1 — Feature Extractor + SVM
─────────────────────────────────────
Image → MobileNetV2 (frozen) → 1280 features → StandardScaler → SVM (Linear, C=1.0) → Label

Approach 2 — End-to-End Fine-tuning
─────────────────────────────────────
Image → MobileNetV2 (last 20 layers unfrozen) → GlobalAvgPool → BN → Dropout(0.4)
      → Dense(128, ReLU) → Dropout(0.3) → Dense(1, Sigmoid) → Label
```

---

## ⚙️ Hyperparameters

| Parameter | Approach 1 | Approach 2A | Approach 2B |
|---|---|---|---|
| MobileNet layers | All frozen | Last 20 unfrozen | Last 20 unfrozen |
| Validation set | 1,043 images | **16 images** | **1,043 images** |
| SVM kernel | Linear | — | — |
| SVM C | 1.0 | — | — |
| Optimizer | — | Adam | Adam |
| Warmup LR | — | 1e-3 | 1e-3 |
| Fine-tune LR | — | 1e-4 | **5e-5** |
| Warmup epochs | — | 10 (max) | 10 (max) |
| Fine-tune epochs | — | 20 (max) | 20 (max) |
| Dropout | — | 0.4 | 0.4 |
| Class weights | — | None | Balanced |
| Batch size | 32 | 32 | 32 |

---

## 🔑 Key Finding: Why Approach 2A Failed

The original Kaggle validation folder contains only **16 images (8 NORMAL + 8 PNEUMONIA)**.

```
Problem:  Early Stopping uses val_accuracy to decide when to stop.
          With only 16 images, 1 wrong prediction = 6.25% accuracy drop.
          The signal is too noisy → model stops at wrong time.

Result:   Model collapsed to predict PNEUMONIA for everything.
          NORMAL recall = 0%  →  Accuracy dropped to ~62-70%

Fix (2B): Use 20% of training data as validation = 1,043 images.
          Early Stopping now has a reliable signal.
          Accuracy recovered to 87.82%
```

---

## 📊 Results Summary

| Metric | Approach 1 (SVM) | Approach 2A (val=16) | Approach 2B (val=1043) |
|---|:---:|:---:|:---:|
| **Accuracy** | 85.10% | 70.51% | **87.82%** |
| **Precision** | 0.8322 | 0.6801 | **0.8520** |
| **Recall** | 0.9538 | 0.9974 | 0.9744 |
| **F1-Score** | 0.8889 | 0.8087 | **0.9091** |
| **AUC** | 0.9415 | 0.9249 | **0.9644** |
| **Train Time** | **8s** | 1,805s | 3,190s |

### Charts (see members/Matthew_xray/results/)

| File | Description |
|---|---|
| `req1_sample_images.png` | Sample X-Ray images per class |
| `req1_validation_comparison.png` | Val set size: 16 vs 1043 images |
| `req1_class_distribution.png` | Class distribution bar chart |
| `req3_cm_approach1.png` | Confusion matrix — Approach 1 (SVM) |
| `req3_cm_approach2a.png` | Confusion matrix — Approach 2A (original) |
| `req3_cm_approach2b.png` | Confusion matrix — Approach 2B (improved) |
| `req3_learning_curves_2a_vs_2b.png` | Learning curves: 2A vs 2B |
| `req4_3way_comparison.png` | Bar chart: all 3 approaches |
| `req4_before_after_val.png` | Before vs after validation fix |
| `req4_3cm_comparison.png` | 3 confusion matrices side by side |
| `req4_roc_all.png` | ROC curves for all 3 approaches |

---

## 🚀 How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run All Steps
```bash
python main.py --member Matthew_xray
```

### Run Individual Steps
```bash
cd members/Matthew_xray
python dataset_analysis.py    # Step 1: EDA
python preprocessing.py       # Step 2: Augmentation
python feature_extraction.py  # Step 3: Extract features
python ml_classifier.py       # Step 4: Train SVM
python endtoend_dl.py         # Step 5: Train E2E model
python evaluation.py          # Step 6: Compare and plot
```

---

## 📚 References

1. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). **MobileNetV2: Inverted Residuals and Linear Bottlenecks**. *CVPR 2018*.
2. Awajan, A., et al. (2023). **Detection of Pneumonia from CXR Using MobileNet Model**. *Healthcare, MDPI*.
3. Mooney, P. (2018). **Chest X-Ray Images (Pneumonia)**. *Kaggle Dataset*.
4. Cortes, C., & Vapnik, V. (1995). **Support-vector networks**. *Machine Learning, 20(3), 273-297*.

---

## 👥 Team Members

| Member | Dataset | Model |
|---|---|---|
| Matthew Mokhles | Chest X-Ray (Pneumonia) | MobileNetV2 |
| Mark | Chest X-Ray (Pneumonia) | VGG16 |
| Mario | PlantVillage (38 classes) | ResNet50 |
| Veronia | CIFAR-10 | ResNet50 |
