# 🐱🐶 Cat vs Dog — End-to-End MLOps Pipeline

> An end-to-end MLOps pipeline for binary image classification (Cats vs Dogs)
> built for a pet adoption platform — covering model development, packaging,
> CI/CD, deployment, and monitoring.

[![CI — Test, Build & Publish](https://github.com/HARSH6355/Dog-Vs-Cat-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/HARSH6355/Dog-Vs-Cat-Classification/actions/workflows/ci.yml)

---

## 📋 Module Completion Status

| Module | Marks | Status | Description |
|--------|-------|--------|-------------|
| **M1** | 10 | ✅ Done | Model Development & Experiment Tracking |
| **M2** | 10 | ✅ Done | Model Packaging & Containerization (FastAPI + Docker) |
| **M3** | 10 | ✅ Done | CI Pipeline (GitHub Actions + GHCR) |
| **M4** | 10 | ✅ Done | CD Pipeline & Deployment (Docker Compose + Kubernetes) |
| **M5** | 10 | ✅ Done | Monitoring, Logs & Final Submission |

---

## 📁 Project Structure

```
cat-vs-dog-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml                    # M3+M4: CI/CD pipeline (test→build→deploy)
├── configs/
│   └── config.yaml                   # Centralized pipeline configuration
├── data/
│   ├── raw/                          # Raw dataset (DVC tracked)
│   └── processed/                    # Preprocessed splits (DVC tracked)
├── k8s/
│   ├── deployment.yaml               # M4: Kubernetes Deployment (2 replicas)
│   └── service.yaml                  # M4: Kubernetes Service (LoadBalancer)
├── models/
│   ├── best_model.pt                 # Trained model weights (~1.9 MB)
│   ├── training_curves.png           # Loss & accuracy curves
│   ├── confusion_matrix.png          # Evaluation heatmap
│   └── classification_report.txt     # Precision / Recall / F1
├── monitoring/
│   └── performance_report.json       # M5: Post-deploy performance report
├── notebooks/
│   └── 01_data_exploration_and_baseline.ipynb
├── scripts/
│   ├── smoke_test.py                 # M4: Post-deploy smoke tests
│   └── simulate_requests.py          # M5: Performance tracking simulation
├── src/
│   ├── api/
│   │   ├── app.py                    # M2: FastAPI inference service
│   │   └── predict.py                # M2: Inference engine (loads model)
│   ├── data/
│   │   ├── download_dataset.py       # Kaggle dataset downloader
│   │   └── preprocess.py             # Transforms, splits, DataLoaders
│   ├── models/
│   │   ├── baseline_cnn.py           # 3-block CNN architecture
│   │   └── train.py                  # Training loop + MLflow tracking
│   └── utils/
│       └── helpers.py                # Device utils, plots, checkpoints
├── tests/
│   ├── test_model.py                 # 14 model unit tests
│   ├── test_preprocessing.py         # 9 preprocessing unit tests
│   └── test_api.py                   # 24 API unit tests
├── .dockerignore                     # Docker build exclusions
├── .dvcignore                        # DVC exclusions
├── docker-compose.yml                # M4: Dev + prod deployment profiles
├── Dockerfile                        # M2: CPU inference container
├── dvc.yaml                          # M1: DVC pipeline definition
└── requirements.txt                  # Pinned dependencies (CUDA for training)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10, Conda
- Docker Desktop (for container steps)
- Git

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/HARSH6355/Dog-Vs-Cat-Classification.git
cd Dog-Vs-Cat-Classification
conda create -n computer-vision python=3.10
conda activate computer-vision
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python src/data/download_dataset.py
```

### 3. Train the Model (GPU)
```bash
python src/models/train.py
```

### 4. View MLflow Dashboard
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

### 5. Run All Unit Tests
```bash
pytest tests/ -v
# 47 tests — expected: all pass
```

---

## 🤖 M1: Model Development & Experiment Tracking

### Architecture — Baseline CNN
```
Input: (B, 3, 224, 224)
    ↓
ConvBlock 1: Conv(32) → BatchNorm → ReLU → MaxPool  → (B,  32, 112, 112)
ConvBlock 2: Conv(64) → BatchNorm → ReLU → MaxPool  → (B,  64,  56,  56)
ConvBlock 3: Conv(128)→ BatchNorm → ReLU → MaxPool  → (B, 128,  28,  28)
    ↓
GlobalAvgPool                                        → (B, 128)
    ↓
FC(512) → Dropout(0.5) → FC(2)                       → (B, 2)
```

### Experiment Tracking (MLflow)
| Logged Item | Details |
|------------|---------|
| Parameters | epochs, LR, batch_size, optimizer, dropout |
| Metrics    | train/val loss & accuracy (per epoch), test accuracy |
| Artifacts  | `best_model.pt`, confusion matrix, loss curves, classification report |

### Data Versioning (DVC)
```bash
dvc repro        # Re-run pipeline
dvc status       # Check stale stages
```

---

## 🌐 M2: Model Packaging & Containerization

### FastAPI Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info & available routes |
| `/health` | GET | Health check (`model_loaded: true/false`) |
| `/predict` | POST | Upload image → `{label, confidence, probabilities, latency_ms}` |
| `/metrics` | GET | Request count, success rate, average latency |
| `/docs` | GET | Interactive Swagger UI |

### Run API Locally (without Docker)
```bash
conda activate computer-vision
uvicorn src.api.app:app --reload --port 8000
```

### Run API via Docker
```bash
# Build
docker build -t cat-dog-classifier:v1 .

# Run
docker run -p 8000:8000 cat-dog-classifier:v1

# Open Swagger UI
# http://localhost:8000/docs
```

### Test a Prediction (curl)
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/path/to/your/cat_or_dog.jpg"
```

**Example response:**
```json
{
  "label": "dog",
  "confidence": 0.9832,
  "probabilities": {"cat": 0.0168, "dog": 0.9832},
  "latency_ms": 12.4,
  "filename": "my_dog.jpg"
}
```

---

## ⚙️ M3: CI Pipeline (GitHub Actions)

Every push to `main` triggers a **3-job pipeline**:

```
┌──────────────────┐    ┌────────────────────────┐    ┌──────────────────────┐
│  Job 1: test     │───▶│  Job 2: build-and-push  │───▶│  Job 3: deploy       │
│                  │    │                          │    │                      │
│  • Python 3.10   │    │  • Build Docker image    │    │  • Pull from GHCR    │
│  • CPU PyTorch   │    │  • Push to GHCR          │    │  • Start container   │
│  • pytest (47)   │    │                          │    │  • Smoke tests       │
└──────────────────┘    └────────────────────────┘    └──────────────────────┘
```

**Published image:**
```
ghcr.io/harsh6355/dog-vs-cat-classification:latest
```

---

## 🚢 M4: CD Pipeline & Deployment

### Option A — Docker Compose (Simple Local Deploy)
```bash
# Development (builds from local source):
docker compose --profile dev up --build

# Production (pulls from GHCR):
docker compose --profile prod pull
docker compose --profile prod up -d
```

### Option B — Kubernetes (Local cluster with minikube/kind)
```bash
# Start cluster (minikube example)
minikube start

# Deploy
kubectl apply -f k8s/

# Access the service
kubectl port-forward svc/cat-dog-api 8000:80 -n cat-dog
# Open http://localhost:8000/docs
```

### Smoke Tests (Post-Deploy Verification)
```bash
python scripts/smoke_test.py --url http://localhost:8000
```
- Calls `GET /health` — asserts `model_loaded: true`
- Calls `POST /predict` — asserts valid label & confidence
- Exits code 1 on failure (fails CI pipeline automatically)

---

## 📊 M5: Monitoring, Logs & Performance Tracking

### Real-Time Logging
Every `/predict` request is logged with:
```
2026-08-28 21:25:32 [INFO] cat_dog_api — POST /predict — file='dog.jpg' content_type='image/jpeg'
2026-08-28 21:25:32 [INFO] cat_dog_api — Prediction: label=dog confidence=0.9832 latency=12.4ms
```

### Metrics Dashboard (GET /metrics)
```json
{
  "total_requests": 150,
  "successful_requests": 148,
  "failed_requests": 2,
  "success_rate_pct": 98.7,
  "avg_latency_ms": 14.3,
  "recent_latencies_count": 150
}
```

### Post-Deployment Performance Simulation
```bash
# Ensure API is running, then:
python scripts/simulate_requests.py --url http://localhost:8000 --n 30

# With real images:
python scripts/simulate_requests.py --images-dir data/raw/ --n 30

# Report is saved to monitoring/performance_report.json
```

---

## 🔑 Configuration Reference (`configs/config.yaml`)

```yaml
training:
  epochs: 20
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam

model:
  filters: [32, 64, 128]
  fc_units: 512
  dropout_rate: 0.5
  num_classes: 2

dataset:
  image_size: [224, 224]
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
```

---

## 📦 Dataset

- **Source**: [Kaggle — dog-and-cat-classification-dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
- **Size**: 24,998 images (12,499 cats + 12,499 dogs)
- **Preprocessing**: 224×224 RGB, ImageNet normalisation
- **Split**: 80% train / 10% validation / 10% test
- **Augmentation**: horizontal flip, rotation ±15°, color jitter

---

## 📝 Deliverables Checklist

- [x] Git repository with full source code
- [x] `dvc.yaml` — data versioning pipeline
- [x] `Dockerfile` + `docker-compose.yml` — container config
- [x] `.github/workflows/ci.yml` — CI/CD pipeline
- [x] `k8s/deployment.yaml` + `k8s/service.yaml` — Kubernetes manifests
- [x] `models/best_model.pt` — trained model artifact
- [x] `models/training_curves.png`, `confusion_matrix.png` — MLflow artifacts
- [x] `monitoring/performance_report.json` — M5 performance report
- [x] Zip file of all artifacts (create with: `git archive --format=zip HEAD -o submission.zip`)
- [x] Screen recording (≤ 5 min) demonstrating end-to-end workflow
