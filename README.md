# 🐱🐶 Cat vs Dog MLOps Pipeline

> **MLOps Assignment 2** — S1-25_AIMLCZG523 | Binary Image Classification

An end-to-end MLOps pipeline for **binary image classification** (Cats vs Dogs) built for a pet adoption platform.

---

## 📁 Project Structure

```
cat-vs-dog-mlops/
├── configs/
│   └── config.yaml                   # Centralized pipeline configuration
├── data/
│   ├── raw/                          # Raw dataset (DVC tracked)
│   └── processed/                    # Preprocessed splits (DVC tracked)
├── notebooks/
│   └── 01_data_exploration_and_baseline.ipynb   # M1 notebook
├── src/
│   ├── data/
│   │   ├── download_dataset.py       # Smart Kaggle dataset downloader
│   │   └── preprocess.py             # Transforms, splits, DataLoaders
│   ├── models/
│   │   ├── baseline_cnn.py           # 3-block CNN architecture
│   │   └── train.py                  # Training + MLflow tracking
│   └── utils/
│       └── helpers.py                # Device, plots, checkpoints
├── models/                           # Saved artifacts (DVC tracked)
├── tests/                            # Unit tests (M3)
├── mlruns/                           # MLflow tracking data
├── dvc.yaml                          # DVC pipeline definition
├── requirements.txt                  # Pinned dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python src/data/download_dataset.py
```
> The dataset is downloaded to `data/raw/` **once** and skipped on subsequent runs.

### 3. Train the Model
```bash
python src/models/train.py
```

### 4. View MLflow Dashboard
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 5. Or use the Notebook
```bash
jupyter notebook notebooks/01_data_exploration_and_baseline.ipynb
```

---

## 📋 Modules

| Module | Status | Description |
|--------|--------|-------------|
| **M1** | ✅ | Model Development & Experiment Tracking |
| M2 | 🔜 | Model Packaging & Containerization (FastAPI + Docker) |
| M3 | 🔜 | CI Pipeline (GitHub Actions) |
| M4 | 🔜 | CD Pipeline & Deployment (Kubernetes/Docker Compose) |
| M5 | 🔜 | Monitoring & Logging |

---

## 🏗️ Architecture — Baseline CNN

```
Input: (B, 3, 224, 224)
    ↓
ConvBlock 1: Conv(32) → BN → ReLU → MaxPool   → (B, 32,  112, 112)
ConvBlock 2: Conv(64) → BN → ReLU → MaxPool   → (B, 64,   56,  56)
ConvBlock 3: Conv(128)→ BN → ReLU → MaxPool   → (B, 128,  28,  28)
    ↓
GlobalAvgPool                                  → (B, 128)
    ↓
FC(512) → Dropout(0.5) → FC(2)                → (B, 2)
```

---

## 🔬 MLflow Tracking

All experiments are tracked with [MLflow](https://mlflow.org/):

- **Parameters**: epochs, LR, batch size, optimizer, model config
- **Metrics**: train/val loss & accuracy (per epoch), test accuracy
- **Artifacts**: model checkpoint, confusion matrix, loss curves, classification report

---

## 🗂️ DVC Pipeline

```
dvc repro    # Re-run the full pipeline
dvc status   # Check which stages are stale
dvc push     # Push data to remote storage
```

Pipeline stages defined in `dvc.yaml`:
1. `download` → pulls dataset from Kaggle to `data/raw/`
2. `train` → trains model, saves checkpoint to `models/`

---

## ⚙️ Configuration

All settings are centralized in [`configs/config.yaml`](configs/config.yaml):

```yaml
training:
  epochs: 20
  batch_size: 32
  learning_rate: 0.001

model:
  filters: [32, 64, 128]
  fc_units: 512
  dropout_rate: 0.5
```

---

## 📦 Dataset

- **Source**: [Kaggle — bhavikjikadara/dog-and-cat-classification-dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
- **Preprocessing**: 224×224 RGB, normalized with ImageNet stats
- **Split**: 80% train / 10% validation / 10% test
- **Augmentation**: horizontal flip, rotation ±15°, color jitter
