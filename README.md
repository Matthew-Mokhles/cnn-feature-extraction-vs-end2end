# Deep Learning Project — Team Repo
**Course:** CAI3105 / CS460 — Deep Learning &nbsp;|&nbsp; **Deadline:** May 7th, 2026

---

## Team Members

| Alias | Folder | Dataset | Architecture | Status | Run Command |
|---|---|---|---|---|---|
| **Matthew** | `members/Matthew_xray/` | Chest X-Ray | ResNet50 | READY | `python main.py --member matthew` |
| **Mark** | `members/mark_xray/` | Chest X-Ray | VGG16 | WIP (owner: Mark) | — |
| **Mario** | `members/Mario_plantvillage/` | PlantVillage | ResNet50 | WIP (owner: Mario) | — |
| **Veronia** | `members/Veronia_cifar10/` | CIFAR-10 | ResNet50 | WIP (owner: Veronia) | — |

The intended architecture is that every member's folder is **fully standalone**:
each carries its own `config.py`, `utils/{metrics,visualization}.py`, step files,
and writes its outputs into its own local `models/` and `results/` subfolders.
There is no shared `core/` package — pipelines do not import each other.


---

## Directory Structure

```
Ai project 3/
│
├── main.py                            ← Single entry point for everyone
├── requirements.txt
├── .gitignore
│
└── members/                           ← Each person's standalone workspace
    ├── Matthew_xray/                  ← Matthew (ResNet50 + X-Ray)
    │   ├── config.py
    │   ├── run.py
    │   ├── dataset_analysis.py        ← Step 1 (EDA)
    │   ├── preprocessing.py           ← Step 2 (augmentation visualisation)
    │   ├── feature_extraction.py      ← Step 3
    │   ├── ml_classifier.py           ← Step 4
    │   ├── endtoend_dl.py             ← Step 5
    │   ├── evaluation.py              ← Step 6
    │   ├── utils/{metrics.py, visualization.py}
    │   ├── models/                    ← Local, gitignored
    │   └── results/                   ← Plots committed after a run
    │
    ├── mark_xray/                     ← Mark (VGG16 + X-Ray)
    │   ├── config.py, run.py, step files, utils/, models/, results/
    │
    ├── Mario_plantvillage/            ← Mario (ResNet50 + PlantVillage 38 cls)
    │   ├── config.py, run.py, step files, utils/, models/, results/
    │
    └── Veronia_cifar10/               ← Veronia (ResNet50 + CIFAR-10 builtin)
        ├── config.py, run.py, step files, utils/, models/, results/
```

Each folder also expects a local `data/` subfolder for image-directory datasets
(Matthew, Mark, Mario). CIFAR-10 (Veronia) auto-downloads through Keras, no
local images needed.

### Matthew — validation split (important for DL training)

End-to-end training uses `val/` for checkpoints and early stopping. **Very small
validation sets** (e.g. only a handful of images per class) make `val_accuracy`
extremely noisy — prefer **stratified validation with hundreds of images per
class** (about **10–20%** of training), or merge pools and use k-fold.

Optional helper (from repo root, after `cd` into this project folder):

```bash
cd cnn-feature-extraction-vs-end2end
python members/Matthew_xray/tools/split_train_val.py --fraction 0.15 --dry-run
python members/Matthew_xray/tools/split_train_val.py --fraction 0.15 --copy
```

Use `--copy` to keep originals in `train/`, or omit it to **move** files into `val/`.

### Matthew — TensorFlow on Windows (speed)

TensorFlow **2.11+** on **native Windows** typically does **not** use NVIDIA CUDA;
training is **CPU-bound** unless you use **WSL2 + CUDA**, **TensorFlow-DirectML**,
or another supported GPU setup. Long runtimes are expected on CPU.

To reduce console noise from TensorFlow C++ logs, set before running Python, e.g.
`TF_CPP_MIN_LOG_LEVEL=2` (Linux/macOS) or `$env:TF_CPP_MIN_LOG_LEVEL='2'` in PowerShell.
`run.py` sets a quiet default via `setdefault` so you can override if needed.

---

## Quick Start

### 1. Clone & install
```bash
git clone <repo-url>
cd cnn-feature-extraction-vs-end2end
pip install -r requirements.txt
```

### 2. Download your dataset
| Member | Dataset | Source | Place at |
|---|---|---|---|
| Matthew / Mark | Chest X-Ray | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | `members/<name>/data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/` |
| Mario | PlantVillage | [Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) | `members/Mario_plantvillage/data/plant_village/{train,val,test}/<class>/` |
| Veronia | CIFAR-10 | auto-downloads via Keras | — |

### 3. Run your pipeline
```bash
# Matthew (the only currently-runnable pipeline)
python main.py --member matthew --step all   # all 6 steps
python main.py --member matthew --step 1     # EDA only
python main.py --member matthew --step 3     # feature extraction only
```

You can also run a member's `run.py` directly, e.g.:
```bash
python members/Matthew_xray/run.py --step 4
```

> Mark, Mario, and Veronia: once your folder is standalone (mirroring
> `members/Matthew_xray/`), add your alias back to `MEMBER_MAP` in
> `main.py` and remove the WIP marker in the table above.

---

## Step Reference (all members)

| Step | File | Purpose |
|---|---|---|
| 1 | `dataset_analysis.py` | Class counts, sample grid, image-size stats |
| 2 | `preprocessing.py` | Build train/val/test generators + augmentation visual |
| 3 | `feature_extraction.py` | Frozen CNN → `.npy` feature vectors |
| 4 | `ml_classifier.py` | SVM + LR on the extracted features |
| 5 | `endtoend_dl.py` | End-to-end fine-tuning of the CNN |
| 6 | `evaluation.py` | Comparison plots, training-time chart, summary |

---

## Git Rules (Zero Conflict)

1. **Never push** `data/`, `models/`, `*.npy`, `*.h5`, `*.joblib` (gitignored).
2. **Each member only edits** their own `members/<name>/` folder.
3. **Push** your `members/<name>/results/` plots after your pipeline completes.
4. Work on your own branch: `git checkout -b your-name` → PR to `main`.

---

## Evaluation Metrics (all members)
Accuracy · Precision · Recall · F1-Score · Confusion Matrix · Training curves

For **Matthew**, the step-6 comparison table uses **class-weighted** Precision /
Recall / F1 (better summary under imbalance). Per-class detail remains in
`classification_report` and confusion matrices. ML models are fit on **train +
val features**; reported ML scores are evaluated on the **held-out test set** only.
