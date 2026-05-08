# 🌿 PlantVillage Disease Classification — EfficientNet-B0
### CAI3105 / CS460 — Deep Learning | 12th Week Project
**College of Computing and Information Technology — South Valley Campus**  
**Lecturer:** Prof. Nashwa El-Bendary

---

## 📌 Project Overview

This project implements and compares two deep learning strategies for plant disease classification on the **PlantVillage dataset**:

| | Approach | Description |
|---|---|---|
| **Approach-1** | End-to-End DL | EfficientNet-B0 fully fine-tuned as a complete classifier |
| **Approach-2** | DL + SVM | EfficientNet-B0 frozen as a feature extractor → RBF-SVM classifier |

The goal is to evaluate whether a traditional ML classifier fed with deep features can compete with a fully integrated end-to-end deep learning model.

---

## 📁 Project Structure

```
project/
│
├── plantvillage_efficientnet.py     ← Main script (this file)
│
├── plantvillage dataset/            ← Dataset folder
│   ├── color/                       ← Used by the script (38 class folders)
│   ├── grayscale/                   ← Not used
│   └── segmented/                   ← Not used
│
└── results/                         ← Auto-generated after running
    ├── dataset_overview.png         ← Dataset metadata table
    ├── class_distribution.png       ← Bar chart of class sizes
    ├── hyperparameter_table.png     ← Full hyperparameter summary
    ├── learning_curves.png          ← Approach-1 train/val accuracy & loss
    ├── cm_dl.png                    ← Approach-1 confusion matrix
    ├── cm_svm.png                   ← Approach-2 confusion matrix
    ├── metric_comparison.png        ← Side-by-side bar chart
    ├── training_time.png            ← Training time comparison
    ├── per_class_accuracy.png       ← Per-class accuracy (hardest classes)
    ├── classification_report_dl.txt ← Approach-1 full per-class report
    ├── classification_report_svm.txt← Approach-2 full per-class report
    ├── results.json                 ← All numeric results
    ├── efficientnet_b0_model/       ← Saved Keras model (Approach-1)
    └── svm_pipeline.pkl             ← Saved SVM + PCA + Scaler (Approach-2)
```

---

## 🗂️ Dataset

| Property | Value |
|---|---|
| **Name** | PlantVillage |
| **Source** | [Kaggle — abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) |
| **Total Images** | ~55,448 |
| **Classes** | 38 (14 crop species: healthy + diseased) |
| **Image Resolution** | 224 × 224 pixels (RGB) |
| **Subset Used** | Color only |
| **Reference** | Mohanty, Hughes & Salathe, *Frontiers in Plant Science*, 2016 |

### Data Split
| Split | Ratio | Images (approx.) |
|---|---|---|
| Training | 70% | ~38,814 |
| Validation | 15% | ~8,317 |
| Test | 15% | ~8,317 |

> Seed = 42 for full reproducibility.

---

## ⚙️ Requirements

### Python Version
```
Python 3.8 or higher
```

### Install Dependencies
```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy pillow
```

### Full dependency list
| Library | Purpose |
|---|---|
| `tensorflow` | EfficientNet-B0 model, training, data generators |
| `scikit-learn` | SVM, PCA, StandardScaler, all metrics |
| `matplotlib` | All plots and charts |
| `seaborn` | Confusion matrix heatmaps |
| `numpy` | Array operations |
| `pillow` | Image loading support |

---

## 🚀 How to Run

### Step 1 — Set your dataset path
Open `plantvillage_efficientnet.py` and find the `CFG` block at the top.  
Change the `dataset_path` to where your dataset is:

```python
CFG = {
    "dataset_path" : r"D:\semster 8\Deep Learning\assignments\12th_project\plantvillage dataset",
    ...
}
```

> The `r` before the string is important on Windows — it handles backslashes correctly.

### Step 2 — Run the script
```bash
python plantvillage_efficientnet.py
```

### Step 3 — Check results
All outputs are saved automatically to the `results/` folder in the same directory you ran the script from.

---

## 🧠 Model Architecture

### EfficientNet-B0 (shared backbone for both approaches)

```
Input (224×224×3)
     ↓
Stem Conv (3×3, 32 filters)
     ↓
MBConv Stages × 7  (Mobile Inverted Bottleneck + Squeeze-and-Excitation)
     ↓
Top Conv (1×1, 1280 filters)
     ↓
Global Average Pooling  →  1280-D feature vector
     ↓
   [Approach-1]              [Approach-2]
Batch Normalization      PCA (1280 → 256-D)
Dropout (0.5)            StandardScaler
Dense (38, softmax)      RBF-SVM (C=10)
     ↓                        ↓
Predicted Class          Predicted Class
```

**Key specs:** ~5.3M parameters | ImageNet pre-trained | Compound scaling

---

## 🔧 Configuration Reference

All parameters are controlled from the `CFG` dictionary at the top of the script. You never need to edit any other part of the code.

```python
CFG = {
    # ── Dataset ──────────────────────────────────────────────────────────
    "dataset_path"   : "path/to/your/plantvillage dataset",
    "img_size"       : (224, 224),       # EfficientNet-B0 standard input
    "color_channels" : 3,                # RGB
    "batch_size"     : 32,

    # ── Data Splits ───────────────────────────────────────────────────────
    "train_ratio"    : 0.70,
    "val_ratio"      : 0.15,
    "test_ratio"     : 0.15,
    "random_seed"    : 42,

    # ── Approach-1: End-to-End DL ─────────────────────────────────────────
    "dl_epochs"      : 20,               # Phase-2 fine-tune epochs
    "dl_lr"          : 1e-4,             # Phase-1 LR (Phase-2 uses LR/10)
    "dl_dropout"     : 0.5,              # Dropout rate before Dense head
    "dl_optimizer"   : "Adam",
    "dl_loss"        : "categorical_crossentropy",
    "fine_tune_at"   : 100,             # Unfreeze EfficientNet from layer 100

    # ── Approach-2: Feature Extractor + SVM ──────────────────────────────
    "svm_kernel"     : "rbf",            # RBF kernel (best for high-dim features)
    "svm_C"          : 10.0,             # Regularisation strength
    "svm_gamma"      : "scale",          # Kernel coefficient
    "pca_components" : 256,              # Compress 1280-D → 256-D before SVM
}
```

---

## 📋 Training Strategy (Approach-1)

Training uses a **two-phase strategy** to protect ImageNet weights:

### Phase-1 — Head Warm-up (10 epochs)
- EfficientNet-B0 base is **fully frozen**
- Only the new classification head trains
- Learning rate: `1e-4`
- Prevents random gradients from destroying pre-trained weights

### Phase-2 — Fine-tuning (up to 20 epochs)
- Layers from index **100 onward are unfrozen** (136 trainable layers)
- Entire network trains end-to-end
- Learning rate: `1e-5` (10× smaller to avoid catastrophic forgetting)
- Early stopping with `patience=5` (restores best weights)
- ReduceLROnPlateau with `factor=0.5, patience=3`

---

## 🤖 SVM Pipeline (Approach-2)

The frozen EfficientNet-B0 backbone converts every image to a **1280-dimensional vector**. Then:

```
1280-D EfficientNet features
         ↓
   PCA (→ 256-D)          # Retains >95% of variance, speeds up SVM
         ↓
  StandardScaler           # Zero-mean, unit-variance normalisation
         ↓
  RBF-SVM (C=10)          # One-vs-rest multi-class classification
         ↓
   Predicted class
```

**Why RBF-SVM over Logistic Regression?**  
EfficientNet's feature space is highly non-linear. The RBF kernel handles complex decision boundaries better than a linear classifier in 256-dimensional space.

---

## 📊 Output Files Explained

| File | Description |
|---|---|
| `dataset_overview.png` | Table showing dataset metadata (source, size, classes, split) |
| `class_distribution.png` | Bar chart showing how many images exist per class |
| `hyperparameter_table.png` | Side-by-side hyperparameter table for both approaches |
| `learning_curves.png` | Training and validation accuracy/loss across all epochs |
| `cm_dl.png` | Confusion matrix for Approach-1 (colour heatmap, all 38 classes) |
| `cm_svm.png` | Confusion matrix for Approach-2 (colour heatmap, all 38 classes) |
| `metric_comparison.png` | Bar chart comparing Accuracy, Precision, Recall, F1 for both |
| `training_time.png` | Bar chart comparing total training time in minutes |
| `per_class_accuracy.png` | Per-class accuracy for the 15 hardest classes in both approaches |
| `classification_report_dl.txt` | Full sklearn classification report — Approach-1 |
| `classification_report_svm.txt` | Full sklearn classification report — Approach-2 |
| `results.json` | All numeric metrics in JSON (easy to load for the PDF report) |
| `efficientnet_b0_model/` | Full saved Keras model — reload with `tf.keras.models.load_model()` |
| `svm_pipeline.pkl` | Saved SVM + PCA + Scaler — reload with `pickle.load()` |

---

## ♻️ Reloading Saved Models

### Reload Approach-1 (Keras model)
```python
import tensorflow as tf
model = tf.keras.models.load_model("results/efficientnet_b0_model")
predictions = model.predict(your_image_batch)
```

### Reload Approach-2 (SVM pipeline)
```python
import pickle
with open("results/svm_pipeline.pkl", "rb") as f:
    pipe = pickle.load(f)

svm    = pipe["svm"]
pca    = pipe["pca"]
scaler = pipe["scaler"]

# To predict new features:
X_new_reduced = scaler.transform(pca.transform(X_new_features))
predictions   = svm.predict(X_new_reduced)
```

---

## 📈 Expected Results

Based on the PlantVillage literature and EfficientNet-B0 benchmarks:

| Metric | Approach-1 (End-to-End DL) | Approach-2 (DL + SVM) |
|---|---|---|
| **Accuracy** | ~98% | ~96% |
| **Precision** | ~98% | ~96% |
| **Recall** | ~98% | ~96% |
| **F1-Score** | ~98% | ~96% |
| **Train Time** | ~3–5 hours (GPU) | ~30–60 min (GPU + CPU) |
| **Inference** | Slower | ~3× faster |

> Actual results will vary based on your hardware and random seed.

---

## ⚠️ Common Issues & Fixes

| Problem | Fix |
|---|---|
| `FileNotFoundError: Dataset not found` | Check `dataset_path` in CFG — use `r"..."` on Windows |
| `No GPU detected` | Install CUDA + cuDNN, or run on Google Colab with T4 GPU |
| Script is very slow | You are running on CPU — enable GPU for 10–20× speedup |
| `color` folder not found | Script auto-detects; make sure your dataset has `color/` subfolder |
| Out of memory error | Reduce `batch_size` in CFG from 32 to 16 |
| SVM takes too long | Reduce `pca_components` from 256 to 128 |

---

## 📚 References

1. Mohanty, S. P., Hughes, D. P., & Salathe, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. *Frontiers in Plant Science*, 7, 1419.
2. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*.
3. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.
4. Deng, J., et al. (2009). ImageNet: A large-scale hierarchical image database. *CVPR 2009*.

---

## 👥 Course Information

| | |
|---|---|
| **Course** | CAI3105 / CS460 — Deep Learning |
| **Lecturer** | Prof. Nashwa El-Bendary |
| **Deadline** | Thursday, May 7th 2026 (11:55 PM) |
| **Platform** | Moodle LMS |
| **Marks** | 20 Marks |
