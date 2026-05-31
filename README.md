# CardioScan AI

**Automated detection of cardiac abnormalities from 12-lead ECG recordings using multimodal deep learning.**

This project classifies ECGs into five diagnostic categories — Normal (NORM), Myocardial Infarction (MI), ST/T Changes (STTC), Conduction Disturbance (CD), and Ventricular Hypertrophy (HYP) — by fusing three data modalities: raw ECG waveform, patient metadata, and clinical text reports. A FastAPI web application (CardioScan AI) serves the trained model as a patient-facing diagnostic portal.

This is an integrated university project covering four modules: **Deep Learning**, **Multimodal Data Analysis**, **Object-Oriented Programming**, and **Computer Networks and Communication in Medicine**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Model Architecture](#model-architecture)
- [Experiments & Results](#experiments--results)
- [Web Application](#web-application)
- [How to Run — Training (Google Colab)](#how-to-run--training-google-colab)
- [How to Run — Web App (PyCharm / Local)](#how-to-run--web-app-pycharm--local)
- [Environment Variables](#environment-variables)
- [Dependencies](#dependencies)
- [Report](#report)
- [References](#references)

---

## Project Overview

CardioScan AI is a complete end-to-end pipeline:

1. **Raw ECG signals** (12-lead, 1000 timesteps at 100Hz) are preprocessed and passed through a 1D-CNN residual network.
2. **Patient metadata** (age, sex, BMI, height, weight, ECG statistics) are processed by a dense network.
3. **Clinical text reports** (German → English translated, tokenised) are processed by a Bidirectional LSTM.
4. A **gated fusion module** dynamically weights the three modalities per sample.
5. A **hierarchical classifier** first separates Normal from Abnormal, then classifies each Abnormal into MI/STTC/CD/HYP.
6. The trained model is deployed via a **FastAPI web app** with patient authentication, history tracking, and a full HTML frontend.

---

## Dataset

**PTB-XL** — a large publicly available 12-lead ECG dataset.

| Property | Value |
|---|---|
| Total recordings | 21,799 |
| Patients | 18,869 |
| Sampling rate used | 100 Hz |
| Signal length | 10 seconds (1,000 timesteps) |
| Leads | 12 |
| Labels | Five diagnostic superclasses |
| Source | Physikalisch-Technische Bundesanstalt, Berlin |

Download PTB-XL from: https://physionet.org/content/ptb-xl/1.0.3/

> The dataset is **not included** in this repository due to size. Download it separately and place it at the path expected by the notebooks (see instructions below).

**Class distribution (after single-label priority conversion):**

| Class | Training samples | Test samples |
|---|---|---|
| NORM | ~7,577 | 952 |
| MI | ~4,379 | 550 |
| STTC | ~3,131 | 382 |
| CD | ~1,915 | 258 |
| HYP | ~416 | 56 |

---

## Repository Structure

```
cardioscan-ai/
├── notebooks/
│   ├── 01_main_hierarchical_model.ipynb    # Main model: gated fusion + hierarchical classifier
│   ├── 02_smote_experiment.ipynb           # Experiment: SMOTE balancing
│   ├── 03_direct_5class_experiment.ipynb   # Experiment: single-stage 5-class softmax
│   ├── 04_ribeiro_experiment.ipynb         # Experiment: Ribeiro pretrained features
│   └── 05_ablation_study.ipynb             # Ablation: ECG+Meta, ECG+Text, ECG only
│
├── ecg_app/                                # FastAPI web application
│   ├── controllers/
│   │   └── application_controller.py       # Orchestrates preprocessing + model
│   ├── ecg_model_hierarchical/             # Saved TensorFlow models (see note below)
│   │   ├── model_binary_v6/
│   │   └── model_abn_v6/
│   ├── models/
│   │   └── multimodal_model.py             # Loads and runs TF SavedModels
│   ├── processors/
│   │   ├── ecg_processor.py                # ECG CSV validation + normalisation
│   │   ├── metadata_processor.py           # Metadata feature engineering
│   │   └── text_processor.py               # Tokenisation + padding
│   ├── templates/
│   │   ├── home.html                       # Landing page
│   │   ├── analyze.html                    # ECG upload form
│   │   ├── results.html                    # Prediction output
│   │   ├── login.html                      # Patient portal login
│   │   ├── dashboard.html                  # Patient overview
│   │   └── reports.html                    # Report history
│   ├── main.py                             # FastAPI app + all routes
│   ├── database.py                         # SQLite schema + queries
│   └── requirements.txt                    # Python dependencies
│
├── report/
│   └── CardioScan_AI_Report.docx           # Full project report
│
├── .env.example                            # Template for environment variables
├── .gitignore
└── README.md
```

> **Note on saved models:** TensorFlow SavedModel folders can exceed GitHub's 100MB file limit. If the `ecg_model_hierarchical/` folder is not included here, download the models from [Google Drive link here] and place them inside `ecg_app/ecg_model_hierarchical/`.

---

## Model Architecture

### Multimodal Gated Fusion Network

```
ECG (1000×12)          Metadata (9,)          Text (100,)
     │                      │                      │
  1D-CNN                Dense(64)×2         Embedding(128)
  Residual ×3                │              BiLSTM(64)
  GAP → Dense(128)     BN+Dropout(0.4)      Dense(64)
     │                      │                      │
  Gate(128) ──────────  Gate(128) ──────────  Gate(128)
     │  (sigmoid)            │  (sigmoid)           │  (sigmoid)
  Hadamard             Hadamard               Hadamard
     └──────────────────────┴──────────────────────┘
                        Concat (384,)
                        Dense(256) + BN + Dropout(0.4)
                        Dense(64) + Dropout(0.3)
                             │
               ┌─────────────┴─────────────┐
         sigmoid(1)                    softmax(4)
       Binary Model                 Abnormal Model
     (NORM vs ABNORM)           (MI / STTC / CD / HYP)
```

### Hierarchical Classification Pipeline

```
Input ECG + Metadata + Text
         │
    Binary Model
         │
    p(ABNORMAL)
    /           \
  < 0.5         ≥ 0.5
  Predict        Abnormal Model
  NORM           + class boost vector
                 argmax → MI / STTC / CD / HYP
```

---

## Experiments & Results

All experiments evaluated on the same hold-out test set (n = 2,198, strat_fold 10).

| Experiment | Accuracy | Weighted F1 | Macro F1 | MI F1 | STTC F1 | CD F1 | HYP F1 |
|---|---|---|---|---|---|---|---|
| Main: Hierarchical + Manual Balance | 0.62 | 0.62 | 0.47 | 0.56 | 0.54 | 0.42 | 0.07 |
| Exp 1: Hierarchical + SMOTE | 0.60 | 0.61 | 0.45 | 0.58 | 0.46 | 0.41 | 0.05 |
| Exp 2: Direct 5-Class (Custom CNN) | 0.65 | 0.65 | 0.51 | 0.64 | 0.58 | 0.52 | 0.08 |
| Exp 3: Ribeiro Pretrained Features | 0.60 | 0.58 | 0.43 | 0.41 | 0.50 | 0.37 | 0.09 |
| Ablation: ECG + Metadata only | ~0.63 | 0.64 | ~0.50 | 0.62 | 0.56 | 0.42 | 0.10 |
| Ablation: ECG + Text only | 0.58 | 0.59 | 0.45 | 0.56 | 0.44 | 0.43 | 0.07 |

**Key findings:**
- Direct 5-class classification achieved the best overall accuracy (65%) without any hierarchical complexity.
- The Ribeiro pretrained model underperformed despite large-scale pretraining, due to the 75% zero-padding artifact required to match its input shape.
- ECG + Metadata is the best modality pair — text is too sparse (only 2.3% of records have genuine reports) to help reliably.
- HYP is the persistent bottleneck across all experiments due to severe class imbalance (416 training samples, 56 test samples).

---

## Web Application

**CardioScan AI** is a FastAPI web app providing AI-powered ECG diagnosis.

### Features
- Upload a 12-lead ECG CSV file (1000 × 12 format)
- Enter patient demographics (age, sex, height, weight) and optional clinical notes
- Receive a per-class probability breakdown and risk level (Low / Moderate / High)
- Patient portal with login, history dashboard, and past report tracking
- Session-based authentication with 8-hour token expiry

### Pages

| Page | URL | Description |
|---|---|---|
| Home | `/` | Landing page with features overview |
| Analysis | `/analyze` | ECG upload form |
| Results | `/results` | Prediction output with ECG preview |
| Login | `/login` | Patient portal authentication |
| Dashboard | `/dashboard` | Personal health overview |
| Reports | `/reports` | Historical report table |

### Demo credentials
```
Email:    patient@demo.com
Password: Cardio123
```

---

## How to Run — Training (Google Colab)

### Step 1 — Upload dataset to Google Drive

1. Download PTB-XL from https://physionet.org/content/ptb-xl/1.0.3/
2. Upload the entire folder to your Google Drive at: `MyDrive/ECG signals/`

### Step 2 — Open a notebook in Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click `File → Upload notebook` and select one of the `.ipynb` files from `notebooks/`

### Step 3 — Mount Drive and run

The first cell of each notebook mounts Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

Then run all cells in order (`Runtime → Run all`). Each notebook is self-contained.

### Step 4 — Download the trained models

After training completes, download the SavedModel folders from Colab:
```python
from google.colab import files
import shutil
shutil.make_archive('model_binary_v6', 'zip', 'model_binary_v6')
files.download('model_binary_v6.zip')
```

Unzip and place inside `ecg_app/ecg_model_hierarchical/`.

### Expected training times (Google Colab T4 GPU)

| Model | Approximate time |
|---|---|
| Binary model (15 epochs) | ~20 minutes |
| Abnormal classifier (15 epochs) | ~25 minutes |
| Ribeiro feature extraction | ~30 minutes |

---

## How to Run — Web App (PyCharm / Local)

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/cardioscan-ai.git
cd cardioscan-ai/ecg_app
```

### Step 2 — Create a virtual environment

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Mac/Linux:
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Set up environment variables

Copy the example file and fill in your values:
```bash
cp .env.example .env
```

Open `.env` and set:
```
SECRET_KEY=any_long_random_string_here
DEMO_EMAIL=patient@demo.com
DEMO_PASSWORD=Cardio123
SESSION_TTL_HOURS=8
```

### Step 5 — Place the trained models

Make sure your folder structure looks like:
```
ecg_app/
└── ecg_model_hierarchical/
    ├── model_binary_v6/
    │   ├── saved_model.pb
    │   └── variables/
    ├── model_abn_v6/
    │   ├── saved_model.pb
    │   └── variables/
    └── tokenizer_v6.pkl
```

### Step 6 — Run the app

```bash
cd ecg_app
uvicorn main:app --reload
```

Then open your browser at: **http://localhost:8000**

### Step 7 — Test with a sample ECG

The model expects a CSV file with shape `(1000, 12)` — 1000 rows (timesteps) and 12 columns (leads), comma-separated, no header.

You can export a test ECG from PTB-XL using the notebooks, or generate a dummy one:
```python
import numpy as np
dummy = np.random.randn(1000, 12)
np.savetxt('test_ecg.csv', dummy, delimiter=',')
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `SECRET_KEY` | Used for session token generation | `fallback-dev-key` |
| `DEMO_EMAIL` | Login email for demo user | — |
| `DEMO_PASSWORD` | Login password for demo user | — |
| `SESSION_TTL_HOURS` | How long sessions stay valid | `8` |

> Never commit your real `.env` file. Only `.env.example` is safe to push.

---

## Dependencies

### Training (Colab notebooks)
```
tensorflow >= 2.12
numpy
pandas
wfdb
scikit-learn
imbalanced-learn
transformers
sentencepiece
nltk
matplotlib
seaborn
lime
```

### Web App
```
fastapi
uvicorn
python-dotenv
tensorflow
numpy
pandas
```

Install all web app dependencies:
```bash
pip install -r ecg_app/requirements.txt
```

---

## Report

The full project report is available in `report/CardioScan_AI_Report.docx`.

It covers:
- Full methodology (label engineering, preprocessing, model architecture, training)
- All experimental results with confusion matrices and classification reports
- Modality ablation study
- LIME explainability analysis
- OOP architecture walkthrough
- FastAPI web app design and API documentation
- Discussion and comparison with published literature

---

## References

1. Wagner et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*. https://doi.org/10.1038/s41597-020-0495-6
2. Strodthoff et al. (2021). Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL. *IEEE JBHI*. https://pmc.ncbi.nlm.nih.gov/articles/PMC8469424
3. Ali Mehdi & Drigh (2026). ECG Classification on PTB-XL: A Data-Centric Approach with Simplified CNN-VAE. arXiv:2603.07558.
4. Ribeiro et al. (2020). Automatic diagnosis of the 12-lead ECG using a deep neural network. *Nature Communications*. https://github.com/antonior92/automatic-ecg-diagnosis
5. Ribeiro, M.T. et al. (2016). 'Why Should I Trust You?': Explaining the Predictions of Any Classifier. KDD 2016.
6. Chawla et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*.

---

*This project is for research and educational purposes only. It is not a substitute for clinical diagnosis by a licensed physician.*
