# 🔐 Network Security — Phishing Detection System (NSPDS)

> **Classifying phishing vs legitimate URLs from 30 engineered web-security features — with a production-grade 4-stage MLOps pipeline, MongoDB data source, DagsHub/MLflow experiment tracking, S3 artifact sync, and FastAPI serving**
>
> An end-to-end network security ML system: phishing data ingested from MongoDB Atlas → schema-validated with KS drift detection → KNN-imputed and transformed → 5 models compared via GridSearchCV → best model tracked in MLflow → served via FastAPI with CI/CD to AWS ECR/EC2.

---

<div align="center">

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Model-RandomForest%20%7C%20GradientBoosting%20%7C%20AdaBoost-orange)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/Tracking-MLflow%20%2B%20DagsHub-0194E2)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![MongoDB](https://img.shields.io/badge/Database-MongoDB%20Atlas-47A248?logo=mongodb)](https://www.mongodb.com/)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20ECR%20%7C%20EC2-orange?logo=amazonaws)](https://aws.amazon.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>


## 📊 Project Slides

👉 **[View the Project Presentation (PPTX)](https://docs.google.com/presentation/d/1RsWFVcLWYyngkRvsV5kNDGGt3b8F5gVO/edit?usp=sharing&rtpof=true&sd=true)**

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [Problem Statement](#1-problem-statement) |
| 2 | [Project Overview](#2-project-overview) |
| 3 | [Tech Stack](#3-tech-stack) |
| 4 | [High-Level Architecture](#4-high-level-architecture) |
| 5 | [Repository Structure](#5-repository-structure) |
| 6 | [Dataset & Features](#6-dataset--features) |
| 7 | [4-Stage MLOps Pipeline](#8-4-stage-mlops-pipeline) |
| 8 | [Model Training & Experiment Tracking](#9-model-training--experiment-tracking) |
| 9 | [FastAPI Serving](#10-fastapi-serving) |
| 10 | [CI/CD & Cloud Deployment](#11-cicd--cloud-deployment) |
| 11 | [How to Replicate — Full Setup Guide](#12-how-to-replicate--full-setup-guide) |
| 12 | [Business Applications & Other Domains](#13-business-applications--other-domains) |
| 13 | [How to Improve This Project](#14-how-to-improve-this-project) |
| 14 | [Troubleshooting](#15-troubleshooting) |
| 15 | [Glossary](#16-glossary) |

---

## 1. Problem Statement

### What problem are we solving?

Phishing attacks — where malicious websites impersonate legitimate ones to steal credentials — remain the most common vector for cybercrime worldwide. Manual inspection of URLs and web page properties is impossible at scale; email security gateways and browsers need automated classifiers that can evaluate hundreds of features in milliseconds.

Phishing URLs exhibit measurable structural and behavioural patterns: they use IP addresses instead of domain names, exploit URL shorteners, lack valid SSL certificates, have recently-registered domains, and generate unusual traffic patterns. An ML classifier trained on these features can flag malicious URLs with high precision in real time.

### What does NSPDS answer?

> *"Given 30 URL and web-page security features — is this URL phishing (1) or legitimate (0)?"*

### Objectives

1. Ingest phishing detection data from MongoDB Atlas and export to a versioned feature store
2. Validate data quality: schema column count check + Kolmogorov-Smirnov drift detection between train and test sets
3. Transform features: KNN Imputation (k=3) to handle missing values → save preprocessor pipeline
4. Compare 5 classifiers via GridSearchCV, select the best by R² score, track with MLflow on DagsHub
5. Sync all artifacts and the final model to AWS S3
6. Serve training and prediction via FastAPI with a CSV upload interface

---

## 2. Project Overview

| Aspect | Detail |
|--------|--------|
| **Dataset** | Phishing website dataset — 11,055 rows, 31 columns (30 features + target) |
| **Target** | `Result` — phishing: −1 → encoded as 0 · legitimate: 1 → kept as 1 |
| **Features** | 30 integer-encoded URL/web security signals (see Section 6) |
| **Data source** | MongoDB Atlas collection (`NetworkData`) |
| **Preprocessing** | KNNImputer (k=3, uniform weights) — handles missing feature values |
| **Drift detection** | Kolmogorov-Smirnov test per column (threshold p=0.05) |
| **Models compared** | Random Forest · Decision Tree · Gradient Boosting · Logistic Regression · AdaBoost |
| **HPT** | GridSearchCV (cv=3) per model on defined parameter grids |
| **Selection metric** | R² score on test set (best model wins) |
| **Experiment tracking** | MLflow → DagsHub remote (logs f1, precision, recall per run) |
| **Artifact storage** | AWS S3 (`netwworksecurity` bucket) — all stages + final model |
| **Serving** | FastAPI: GET `/train` triggers full pipeline · POST `/predict` accepts CSV |
| **CI/CD** | GitHub Actions → Docker → AWS ECR → EC2 (self-hosted runner) |

---

## 3. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.10 | Core language |
| **Data source** | MongoDB Atlas + `pymongo` | Phishing dataset storage and retrieval |
| **ML framework** | Scikit-learn | 5 classifiers, GridSearchCV, KNNImputer, Pipeline |
| **Drift detection** | `scipy.stats.ks_2samp` | Kolmogorov-Smirnov test for train/test distribution drift |
| **Experiment tracking** | MLflow 2.x + DagsHub | Logs classification metrics and registers best model |
| **Model serialisation** | pickle (stdlib) | Saves preprocessor.pkl + model.pkl |
| **Config management** | PyYAML | Schema validation via `data_schema/schema.yaml` |
| **Env management** | `python-dotenv` | Loads `MONGODB_URL_KEY` / `MONGO_DB_URL` from `.env` |
| **Web framework** | FastAPI + Uvicorn | Serves `/train` and `/predict` routes |
| **Templating** | Jinja2 | Renders prediction results as HTML table |
| **Cloud storage** | AWS S3 (`aws s3 sync`) | Artifact and model versioning |
| **Containerisation** | Docker (`python:3.10-slim-buster`) | Reproducible deployment |
| **CI/CD** | GitHub Actions (3-job pipeline) | Build → push to ECR → deploy on EC2 |
| **Container registry** | AWS ECR | Stores Docker images |
| **Logging** | Python `logging` | Timestamped per-run log files in `logs/` |
| **Packaging** | `setup.py` + `setuptools` | Installs `networksecurity` as importable package |

---

## 4. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                    │
│                                                                      │
│  MongoDB Atlas                                                       │
│  Database: <your_database_name>                                      │
│  Collection: NetworkData                                             │
│         │                                                            │
│  push_data.py — converts phisingData.csv → JSON → inserts to Mongo  │
└──────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│              4-STAGE MLOPS PIPELINE (TrainingPipeline)               │
│                                                                      │
│  Stage 1: DataIngestion                                              │
│    pymongo.find() → DataFrame → feature_store/ → train/test split   │
│                                                                      │
│  Stage 2: DataValidation                                             │
│    Schema column count check                                         │
│    KS drift test (p < 0.05 = drift) per column → drift_report.yaml  │
│                                                                      │
│  Stage 3: DataTransformation                                         │
│    KNNImputer(k=3, uniform) → sklearn Pipeline                       │
│    Result: −1 → 0 (phishing), 1 → 1 (legitimate)                    │
│    → train.npy + test.npy + preprocessing.pkl                        │
│                                                                      │
│  Stage 4: ModelTrainer                                               │
│    GridSearchCV across 5 classifiers → best by R² score             │
│    MLflow → DagsHub: log f1 + precision + recall                    │
│    → final_model/model.pkl + final_model/preprocessor.pkl           │
│                                                                      │
│  S3 Sync: Artifacts/ → s3://netwworksecurity/artifact/{timestamp}   │
│           final_model/ → s3://netwworksecurity/final_model/{ts}     │
└──────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────────┐
│                      FASTAPI SERVING                                 │
│                                                                      │
│  GET  /train   → TrainingPipeline.run_pipeline()                    │
│  POST /predict → CSV upload → NetworkModel.predict() →              │
│                  HTML table with predicted_column                    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
              GitHub push → Actions → Docker → ECR → EC2
```

---

## 5. Repository Structure

```
networksecurity/
│
├── networksecurity/                    # Core Python package
│   ├── cloud/
│   │   └── s3_syncer.py               # S3Sync — aws s3 sync wrapper
│   ├── components/
│   │   ├── data_ingestion.py          # DataIngestion — MongoDB → feature store → train/test split
│   │   ├── data_validation.py         # DataValidation — schema check + KS drift detection
│   │   ├── data_transformation.py     # DataTransformation — KNNImputer + .npy export
│   │   └── model_trainer.py           # ModelTrainer — GridSearchCV + MLflow tracking
│   ├── constant/
│   │   └── training_pipeline/__init__.py  # All pipeline constants (TARGET, paths, thresholds)
│   ├── entity/
│   │   ├── config_entity.py           # Config dataclasses per pipeline stage
│   │   └── artifact_entity.py         # Artifact dataclasses (typed outputs per stage)
│   ├── exception/
│   │   └── exception.py               # NetworkSecurityException (file + line traceback)
│   ├── logging/
│   │   └── logger.py                  # Timestamped log file per run in logs/
│   ├── pipeline/
│   │   └── training_pipeline.py       # TrainingPipeline — orchestrates all 4 stages + S3 sync
│   └── utils/
│       ├── main_utils/utils.py        # read/write YAML, save/load numpy, pickle, evaluate_models()
│       └── ml_utils/
│           ├── metric/classification_metric.py  # get_classification_score() → F1/P/R artifact
│           └── model/estimator.py     # NetworkModel(preprocessor, model) → predict()
│
├── data_schema/
│   └── schema.yaml                    # Column definitions + numerical_columns list
│
├── Network_Data/
│   └── phisingData.csv                # Source dataset (11,055 rows)
│
├── final_model/                       # Production model artifacts
│   ├── model.pkl                      # Best trained classifier
│   └── preprocessor.pkl               # Fitted KNNImputer pipeline
│
├── Artifacts/                         # Timestamped pipeline run outputs (DVC-style)
│
├── templates/
│   └── table.html                     # Jinja2 template for prediction results
│
├── app.py                             # FastAPI application entry point
├── main.py                            # Manual pipeline runner (no API)
├── push_data.py                       # CSV → MongoDB loader
├── test_mongodb.py                    # MongoDB connection test
├── Dockerfile                         # python:3.10-slim-buster + awscli
├── .github/workflows/main.yml         # GitHub Actions CI/CD
├── data_schema/schema.yaml            # Feature schema
├── requirements.txt                   # All dependencies
└── setup.py                           # Package: NetworkSecurity
```

---

## 6. Dataset & Features

### Dataset

| Property | Detail |
|----------|--------|
| **File** | `Network_Data/phisingData.csv` |
| **Rows** | 11,055 website records |
| **Columns** | 31 (30 features + `Result` target) |
| **Feature encoding** | Integer: typically `−1` (suspicious/phishing) · `0` (uncertain) · `1` (legitimate) |
| **Target** | `Result`: `−1` (phishing) → recoded as `0` · `1` (legitimate) → kept as `1` |
| **Missing values** | Possible — handled by KNNImputer (k=3) |

### Feature Groups

The 30 features capture URL structure, SSL/domain trust, page content signals, and external metrics:

| Group | Features | What they capture |
|-------|---------|-------------------|
| **URL structure** | `having_IP_Address`, `URL_Length`, `Shortining_Service`, `having_At_Symbol`, `double_slash_redirecting`, `Prefix_Suffix` | Whether URL uses raw IP, is abnormally long, uses shorteners, or has suspicious syntax |
| **Domain trust** | `having_Sub_Domain`, `Domain_registeration_length`, `age_of_domain`, `DNSRecord` | Domain age, registration duration, and DNS record presence |
| **SSL & protocol** | `SSLfinal_State`, `HTTPS_token`, `Favicon` | Certificate validity, HTTPS usage, favicon origin |
| **Network/port** | `port` | Whether the URL uses non-standard ports |
| **Page content** | `Request_URL`, `URL_of_Anchor`, `Links_in_tags`, `SFH`, `Submitting_to_email`, `Abnormal_URL`, `Iframe`, `popUpWidnow`, `RightClick`, `on_mouseover`, `Redirect` | Behavioural signals: form actions, embedded links, pop-ups, right-click blocking |
| **External signals** | `web_traffic`, `Page_Rank`, `Google_Index`, `Links_pointing_to_page`, `Statistical_report` | Traffic volume, Google indexing, external link counts |

### Target Encoding

The pipeline recodes the target before training:

```python
target_feature_train_df = target_feature_train_df.replace(-1, 0)
# −1 (phishing) → 0
#  1 (legitimate) → 1
```

This converts the −1/1 scheme to the standard 0/1 binary classification format.

---

## 7. 4-Stage MLOps Pipeline

The pipeline is orchestrated by `TrainingPipeline.run_pipeline()`, which calls each stage in sequence and passes typed artifact objects between them.

### Stage 1 — Data Ingestion (`DataIngestion`)

**Config:** `DataIngestionConfig` → paths under `Artifacts/{timestamp}/data_ingestion/`

1. Connect to MongoDB Atlas via `MONGO_DB_URL` env variable
2. `collection.find()` → `pd.DataFrame` → drop `_id` column → replace `"na"` with `np.nan`
3. Export full DataFrame to `feature_store/phisingData.csv`
4. `train_test_split(test_size=0.2)` → write `ingested/train.csv` + `ingested/test.csv`

**Output artifact:** `DataIngestionArtifact(trained_file_path, test_file_path)`

---

### Stage 2 — Data Validation (`DataValidation`)

**Config:** `DataValidationConfig` → `data_validation/validated/` + `drift_report/`

1. Read train and test CSVs
2. **Column count check** — compares against `data_schema/schema.yaml` (31 entries)
3. **Drift detection** — Kolmogorov-Smirnov test per column:
   ```python
   is_same_dist = ks_2samp(d1, d2)
   drift_found = is_same_dist.pvalue < 0.05  # p < threshold = drift
   ```
4. Writes `drift_report/report.yaml` with per-column p-values and drift status
5. Copies validated data to `validated/train.csv` and `validated/test.csv`

**Output artifact:** `DataValidationArtifact(validation_status, valid_train_file_path, valid_test_file_path, drift_report_file_path)`

---

### Stage 3 — Data Transformation (`DataTransformation`)

**Config:** `DataTransformationConfig` → `data_transformation/transformed/` + `transformed_object/`

1. Drop `Result` (target) from features; separate into `X` and `y`
2. Recode target: `y.replace(-1, 0)` — phishing=0, legitimate=1
3. Build sklearn `Pipeline([("imputer", KNNImputer(n_neighbors=3, weights="uniform"))])`
4. Fit on training X only → transform both train and test
5. Concatenate: `np.c_[X_transformed, y_array]` → save as `.npy` files
6. Save preprocessor pipeline: `transformed_object/preprocessing.pkl` AND `final_model/preprocessor.pkl`

**Output artifact:** `DataTransformationArtifact(transformed_object_file_path, transformed_train_file_path, transformed_test_file_path)`

---

### Stage 4 — Model Training (`ModelTrainer`)

**Config:** `ModelTrainerConfig` → `model_trainer/trained_model/`

The trainer benchmarks 5 classifiers with `GridSearchCV(cv=3)`:

| Model | Hyperparameter Grid |
|-------|-------------------|
| `DecisionTreeClassifier` | `criterion`: gini, entropy, log_loss |
| `RandomForestClassifier` | `n_estimators`: 8, 16, 32, 128, 256 |
| `GradientBoostingClassifier` | `learning_rate` × `subsample` × `n_estimators` |
| `LogisticRegression` | (default) |
| `AdaBoostClassifier` | `learning_rate` × `n_estimators` |

**Selection:** Best model by **R² score** on test set → logged to MLflow with F1, precision, recall metrics.

**Saved artifacts:**
- `model_trainer/trained_model/model.pkl` — best classifier
- `final_model/model.pkl` — production copy
- `final_model/preprocessor.pkl` — KNNImputer pipeline

---

### S3 Sync

After training, the full pipeline syncs to S3:

```python
# Artifacts/ → s3://netwworksecurity/artifact/{timestamp}/
self.s3_sync.sync_folder_to_s3(folder=artifact_dir, aws_bucket_url=...)

# final_model/ → s3://netwworksecurity/final_model/{timestamp}/
self.s3_sync.sync_folder_to_s3(folder=model_dir, aws_bucket_url=...)
```

> **Note:** The S3 bucket name `netwworksecurity` contains a typo (extra `w`). Update `TRAINING_BUCKET_NAME` in `constant/training_pipeline/__init__.py` to your own bucket name.

---

## 8. Model Training & Experiment Tracking

### MLflow on DagsHub

Every model evaluation call logs to MLflow:

```python
with mlflow.start_run():
    mlflow.log_metric("f1_score",        f1_score)
    mlflow.log_metric("precision",       precision_score)
    mlflow.log_metric("recall_score",    recall_score)
    mlflow.sklearn.log_model(best_model, "model")
    # If remote store: also registers in MLflow Model Registry
```

`track_mlflow()` is called **twice** — once for training metrics and once for test metrics — giving two runs per model selection, enabling train/test metric comparison in the DagsHub UI.

### Expected Accuracy Threshold

`MODEL_TRAINER_EXPECTED_SCORE = 0.6` — if the best model R² is below this threshold, the pipeline should raise a warning (the current implementation evaluates but does not enforce this; see Section 14).

### Overfitting/Underfitting Threshold

`MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD = 0.05` — the difference between train and test R² scores. Currently defined but not enforced in the training loop.

---

## 9. FastAPI Serving

### Routes

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Redirects to `/docs` (Swagger UI) |
| `GET` | `/train` | Triggers `TrainingPipeline.run_pipeline()` — runs all 4 stages |
| `POST` | `/predict` | Accepts CSV file upload → predicts each row → returns HTML table |

### Prediction Flow

```
POST /predict
  ├── file: UploadFile → pd.read_csv(file.file)
  ├── preprocesor = load_object("final_model/preprocessor.pkl")
  ├── final_model = load_object("final_model/model.pkl")
  ├── network_model = NetworkModel(preprocessor, model)
  ├── y_pred = network_model.predict(df)
  │     ├── preprocessor.transform(x) → KNN-imputed features
  │     └── model.predict(x_transformed) → 0 or 1
  ├── df['predicted_column'] = y_pred
  ├── df.to_csv('prediction_output/output.csv')
  └── Return: templates.TemplateResponse("table.html", ...)
```

The `NetworkModel` wrapper bundles preprocessor + classifier into a single `predict()` call, ensuring the same KNN imputation applied at training is always applied at inference.

### Running the API

```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8000
# Swagger: http://localhost:8000/docs
```

---

## 10. CI/CD & Cloud Deployment

Every push to `main` (excluding `README.md`) triggers the 3-job GitHub Actions pipeline:

```
git push origin main
        │
  Job 1: CI (ubuntu)     Job 2: CD Build (ubuntu)     Job 3: Deploy (self-hosted EC2)
  ─────────────────      ──────────────────────────    ───────────────────────────────
  Lint echo              Configure AWS creds           Configure AWS creds
  Unit test echo         Login to ECR                  Login to ECR
                         docker build                  docker pull ECR:latest
                         docker push → ECR:latest      docker run -d -p 8080:8080
                                                       docker system prune -f
```

### Required GitHub Secrets

| Secret | Value |
|--------|-------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret |
| `AWS_REGION` | e.g. `us-east-1` |
| `AWS_ECR_LOGIN_URI` | e.g. `788614365622.dkr.ecr.us-east-1.amazonaws.com/networkssecurity` |
| `ECR_REPOSITORY_NAME` | ECR repo name |

### Dockerfile

```dockerfile
FROM python:3.10-slim-buster
WORKDIR /app
COPY . /app
RUN apt update -y && apt install awscli -y
RUN apt-get update && pip install -r requirements.txt
CMD ["python3", "app.py"]
```

---

## 11. How to Replicate — Full Setup Guide

### Prerequisites

- Python 3.10+
- MongoDB Atlas account (free tier sufficient)
- AWS account (S3, ECR, EC2)
- DagsHub account (for MLflow tracking)
- Docker Desktop (optional for local)

---

### Step 1 — Clone & Install

```bash
git clone https://github.com/sahatanmoyofficial/Network-Security.git
cd networksecurity

python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate            # Windows

pip install -r requirements.txt
pip install -e .
```

---

### Step 2 — Configure Environment

Create `.env` in project root:

```env
MONGODB_URL_KEY=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
MONGO_DB_URL=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/
```

---

### Step 3 — Push Data to MongoDB

```bash
python push_data.py
# Loads Network_Data/phisingData.csv → converts to JSON → inserts into MongoDB
```

---

### Step 4 — Run the Full Pipeline

```bash
# Option A: via FastAPI
python app.py
# GET http://localhost:8000/train

# Option B: directly
python main.py
```

---

### Step 5 — Monitor with MLflow

```bash
# Local UI
mlflow ui
# http://localhost:5000

# Or view on DagsHub after updating credentials in model_trainer.py
```

---

## 12. Business Applications & Other Domains

### Primary Use Case — URL Phishing Detection

| Stakeholder | Value |
|-------------|-------|
| **Email security gateways** | Scan links in incoming emails — flag phishing URLs before delivery |
| **Browser extensions** | Real-time URL classification as users browse |
| **Enterprise SOC teams** | Bulk-classify URLs from threat intelligence feeds |
| **ISPs / DNS providers** | Block known phishing patterns at the network layer |
| **Financial institutions** | Protect customers from brand-impersonation attacks |

### The Architecture Generalises

The 4-stage MLOps pattern (ingest → validate → transform → train) with MongoDB source and FastAPI serving applies to many classification problems:

| Domain | Classification Task | Feature Analogues |
|--------|--------------------|--------------------|
| **Fraud detection** | Transaction fraud vs legitimate | Amount, frequency, merchant category |
| **Network intrusion** | Attack vs normal traffic | Packet features, connection duration |
| **Spam detection** | Spam vs ham email | Word counts, link density, sender features |
| **Healthcare** | Disease present vs absent | Lab values, vitals, patient history |
| **Credit risk** | Default vs non-default | Financial ratios, payment history |

---

## 13. How to Improve This Project

### 🧠 Model Improvements

| Area | Priority | Recommendation |
|------|----------|---------------|
| **Enforce expected accuracy threshold** | 🔴 High | `MODEL_TRAINER_EXPECTED_SCORE = 0.6` is defined but never enforced — add check: `if best_score < threshold: raise ValueError(...)` |
| **Switch to F1 as selection metric** | 🔴 High | `evaluate_models()` uses R² score, which is a regression metric — replace with `f1_score` or `roc_auc_score` for binary classification |
| **Enforce overfitting threshold** | 🟡 Medium | Train-test score gap check (`abs(train_score - test_score) > 0.05`) is defined but not applied |
| **Add XGBoost / LightGBM** | 🟡 Medium | Both are standard additions to the model comparison grid for tabular classification |
| **Add SMOTE** | 🟡 Medium | Check class balance; if phishing class is underrepresented, add SMOTE before fitting |

### 🏗️ Engineering Improvements

| Area | Recommendation |
|------|---------------|
| **Fix S3 bucket name typo** | `TRAINING_BUCKET_NAME = "netwworksecurity"` has a double-w — correct to your actual bucket name |
| **Unit tests** | Test `get_classification_score()`, `evaluate_models()`, `DataIngestion.export_collection_as_dataframe()` |
| **Input validation in `/predict`** | Validate uploaded CSV has the expected 30 columns before inference |
| **Add `/health` endpoint** | Check FastAPI is running and `final_model/` artifacts exist |
| **Use DVC for data versioning** | Currently no DVC — add DVC with S3 remote to track `phisingData.csv` and model artifacts formally |
| **Fix `batch_prediction.py`** | Currently empty — implement batch CSV inference pipeline |

---

## 14. Troubleshooting

| Error / Symptom | Fix |
|----------------|-----|
| `MONGO_DB_URL` is `None` | Ensure `.env` is in project root and `python-dotenv` is installed; check env variable name matches exactly |
| `NetworkSecurityException: ... collection not found` | Run `push_data.py` first to populate the MongoDB collection before running the pipeline |
| `FileNotFoundError: final_model/model.pkl` | Run the full training pipeline first; the `/predict` endpoint requires trained artifacts |
| `ValueError: R² score below expected` | Best model didn't reach 0.6 threshold — try more epochs in GridSearchCV or add XGBoost |
| S3 sync fails | Ensure `awscli` is configured on the machine; check IAM role has `S3FullAccess` |
| Docker build fails | Ensure `requirements.txt` is in project root; check Python 3.10 compatibility of all packages |
| `ks_2samp` drift detected | Train and test distributions differ significantly — inspect `drift_report.yaml` for drifted columns |

---

## 15. Glossary

| Term | Definition |
|------|-----------|
| **Phishing** | A cyberattack where malicious websites impersonate legitimate ones to steal credentials |
| **KNNImputer** | Scikit-learn imputer that fills missing values using the mean of the k nearest neighbours |
| **KS test** | Kolmogorov-Smirnov test — non-parametric test for whether two samples come from the same distribution; used here for data drift detection |
| **Data drift** | When the statistical distribution of the test set differs significantly from the training set (p < 0.05 here) |
| **MLflow** | Open-source ML lifecycle platform for experiment tracking, model versioning, and deployment |
| **DagsHub** | Git + DVC + MLflow hosting platform providing a remote MLflow tracking server |
| **NetworkModel** | Wrapper class bundling a preprocessor pipeline and a trained classifier into a single `predict()` method |
| **R² score** | Coefficient of determination — used here as model selection metric (note: better replaced by F1 for binary classification) |
| **GridSearchCV** | Exhaustive cross-validated hyperparameter search across a parameter grid |
| **Feature store** | Directory (`feature_store/`) storing the raw ingested dataset snapshot per pipeline run |
| **Artifact** | Typed output of each pipeline stage (dataclass with file paths) passed to the next stage |
| **S3 Sync** | AWS CLI command `aws s3 sync` — synchronises a local directory to an S3 bucket |
| **ECR** | Amazon Elastic Container Registry — stores Docker images for deployment |
| **Self-hosted runner** | GitHub Actions runner running on your own EC2 instance for the deployment job |
| **TRAINING_BUCKET_NAME** | S3 bucket constant in `training_pipeline/__init__.py` — contains a typo (`netwworksecurity`) that needs correcting |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Tanmoy Saha**
[linkedin.com/in/sahatanmoyofficial](https://linkedin.com/in/sahatanmoyofficial) | sahatanmoyofficial@gmail.com
