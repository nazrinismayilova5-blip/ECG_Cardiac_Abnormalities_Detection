# ECG Cardiac Abnormalities Detection

**Automated detection of cardiac abnormalities from 12-lead ECG recordings using multimodal deep learning.**

This project classifies ECGs into five diagnostic categories: Normal (NORM), Myocardial Infarction (MI), ST/T Changes (STTC), Conduction Disturbance (CD), and Ventricular Hypertrophy (HYP). Classification is done by fusing three data modalities: raw ECG waveform, patient metadata, and clinical text reports. A FastAPI web application (CardioScan AI) serves the trained model as a patient-facing diagnostic portal.

This is an integrated university project covering four modules: **Deep Learning**, **Multimodal Data Analysis**, **Object-Oriented Programming**, and **Computer Networks and Communication in Medicine**.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Model Architecture](#model-architecture)
- [Experiments & Results](#experiments--results)
- [Web Application](#web-application)
- [How to Run: Training (Google Colab)](#how-to-run--training-google-colab)
- [How to Run: Web App (PyCharm/Local)](#how-to-run--web-app-pycharm--local)
- [Environment Variables](#environment-variables)
- [Dependencies](#dependencies)
- [Report](#report)
- [References](#references)

---

## Project Overview

CardioScan AI is a complete end-to-end pipeline:

1. **Raw ECG signals** (12-lead, 1000 timesteps at 100Hz) are preprocessed and passed through a 1D-CNN residual network.
2. **Patient metadata** (age, sex, BMI, height, weight, ECG statistics) are processed by a dense network.
3. **Clinical text reports** (German to English translated, tokenised) are processed by a Bidirectional LSTM.
4. A **gated fusion module** dynamically weights the three modalities per sample. 
5. A **hierarchical classifier** first separates Normal from Abnormal, then classifies each Abnormal into MI/STTC/CD/HYP.
6. The trained model is deployed via a **FastAPI web app** with patient authentication, history tracking, and a full HTML frontend.

---

## Dataset

**PTB-XL** is a large publicly available ECG dataset.

| Property | Value |
|---|---|
| Total recordings | 21,799 |
| Patients | 18,869 |
| Sampling rate used | 100 Hz |
| Signal length | 10 seconds (1,000 timesteps) |
| Leads | 12 |
| Labels | Five diagnostic superclasses |

Download PTB-XL from: https://physionet.org/content/ptb-xl/1.0.3/

> The dataset is not included in this repository due to size. Download it separately and place it at the path expected by the notebooks (see instructions below).

**Class distribution (after single-label priority conversion):**

| Class | Number of samples (before balancing) |
|-------|-----------------|
| NORM  | ~9,480          |
| MI    | ~5,469          |
| STTC  | ~3,896          |
| CD    | ~2,417          |
| HYP   | ~537            |

---

## Repository Structure

```
ecg_app/
├── notebooks/
│   ├── Ecg-signal-analysis-model (Hierarchical)    # Main model: gated fusion + hierarchical classifier
│   ├── ECG-signal-analysis-model (direct)   # Experiment: single-stage 5-class softmax
│                            
├── controllers/
│   └── application_controller.py       # Orchestrates preprocessing + model
├── ecg_model_hierarchical/             # Saved TensorFlow models (see note below)
│   ├── model_binary_v6/
│   └── model_abn_v6/
├── models/
│   └── multimodal_model.py             # Loads and runs TF SavedModels
├── processors/
│   ├── ecg_processor.py                # ECG CSV validation + normalisation
│   ├── metadata_processor.py           # Metadata feature engineering
│   └── text_processor.py               # Tokenisation + padding
├── templates/
│   ├── home.html                       # Landing page
│   ├── analyze.html                    # ECG upload form
│   ├── results.html                    # Prediction output
│   ├── login.html                      # Patient portal login
│   ├── dashboard.html                  # Patient overview
│   ├── reports.html                    # Report history
├── main.py                             # FastAPI app + all routes
├── database.py                         # SQLite schema + queries
├──  requirements.txt                   # Python dependencies
│
├── report/
│   └── Detection of Cardiac Abnormalities using ECG Signals.docx           # Full project report
│
├── .env.example                            # Template for environment variables
├── .gitignore
```

---

## Model Architecture

### Multimodal Gated Fusion Network

```
    ECG                 Metadata                 Text 
     │                      │                      │
   1D-CNN                Dense layer         Embedding layer
Residual layer              │                    BiLSTM
Dense layer               Dropout              Dense layer
     │                      │                      │
  Gate      ──────────     Gate     ──────────    Gate
     │  (sigmoid)           │  (sigmoid)           │  (sigmoid)
  Hadamard             Hadamard               Hadamard
     └──────────────────────┴──────────────────────┘
                           Concat 
                   Dense layer + DropOut
                            │
              ┌─────────────┴─────────────┐
           sigmoid                      softmax
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
                 argmax for MI / STTC / CD / HYP
```

---

## Experiments & Results

All experiments evaluated on the same hold-out test set (n = 2,198, strat_fold 10).

| Experiment | Accuracy | Weighted F1 | Macro F1 | MI F1 | STTC F1 | CD F1 | HYP F1 |
|---|---|---|---|---|---|---|---|
| Main: Hierarchical + Manual Balance | 0.62 | 0.62 | 0.47 | 0.56 | 0.54 | 0.42 | 0.07 |
| Exp 1: Hierarchical + SMOTE | 0.60 | 0.61 | 0.45 | 0.58 | 0.46 | 0.41 | 0.05 |
| Exp 2: Direct 5-Class | 0.65 | 0.65 | 0.51 | 0.64 | 0.58 | 0.52 | 0.08 |
| Exp 3: Ribeiro Pretrained Features | 0.60 | 0.58 | 0.43 | 0.41 | 0.50 | 0.37 | 0.09 |
| Ablation: ECG + Metadata only | 0.63 | 0.64 | 0.50 | 0.62 | 0.56 | 0.42 | 0.10 |
| Ablation: ECG + Text only | 0.58 | 0.59 | 0.45 | 0.56 | 0.44 | 0.43 | 0.07 |
| Ablation: ECG only | 0.61 | 0.62 | 0.48 | 0.61 | 0.52 | 0.44 | 0.07 |

**Key findings:**
- Direct 5-class classification achieved the best overall accuracy (65%) without any hierarchical complexity.
- The Ribeiro pretrained model underperformed despite large-scale pretraining, due to the 75% zero-padding artifact required to match its input shape.
- ECG + Metadata is the best modality pair, text is too sparse to help reliably.
- HYP is the persistent bottleneck across all experiments due to severe class imbalance.

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

## How to Run the Training (Google Colab)

### Step 1: Upload dataset to Google Drive

1. Download PTB-XL from https://physionet.org/content/ptb-xl/1.0.3/
2. Upload the entire folder to your Google Drive at: `MyDrive/ECG signals/`

### Step 2: Open a notebook in Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click `File -> Upload notebook` and select one of the `.ipynb` files from `notebooks/`

### Step 3: Mount Drive and run

The first cell of each notebook mounts Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

Then run all cells in order (`Runtime -> Run all`). Each notebook is self-contained.

### Step 4: Download the trained models

After training completes, download the SavedModel folders from Colab:
```python
from google.colab import files
import shutil
shutil.make_archive('model_binary_v6', 'zip', 'model_binary_v6')
files.download('model_binary_v6.zip')
```

Unzip and place inside `ecg_app/ecg_model_hierarchical/`.

---

## How to Run: Web App (PyCharm/Local)

### Step 1: Clone the repository

### Step 2: Create a virtual environment

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Mac/Linux:
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set up environment variables

Copy the example file and fill in your values:
```bash
cp .env.example .env
```

Open `.env` and set:
```
DEMO_EMAIL=patient@demo.com
DEMO_PASSWORD=Cardio123
```

### Step 5: Place the trained models

Make sure your folder structure looks like:
```
ecg_app/
└── ecg_model_hierarchical/
    ├── model_binary_v6/
    ├── model_abn_v6/
    └── tokenizer_v6.pkl
```

### Step 6: Run the app

```bash
cd ecg_app
uvicorn main:app --reload
```

Then open your browser at: **http://localhost:8000**

### Step 7: Test with a sample ECG

The model expects a CSV file with shape `(1000, 12)` with 1000 rows (timesteps) and 12 columns (leads), comma-separated, no header.

You can export a test ECG from PTB-XL using the notebooks, or generate a dummy one:
```python
import numpy as np
dummy = np.random.randn(1000, 12)
np.savetxt('test_ecg.csv', dummy, delimiter=',')
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| DEMO_EMAIL | Login email for demo user |
| DEMO_PASSWORD | Login password for demo user |

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

The full project report is available in `report/
Detection of Cardiac Abnormalities using ECG Signals
.docx`.

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

1)	Pratima Mishra, 23.07.2025, Z-Score Normalization: Definition and Examples
https://www.geeksforgeeks.org/data-analysis/z-score-normalization-definition-and-examples/
2)	Emergent Mind, 29.10.2025, Gated Fusion Mechanisms https://www.emergentmind.com/topics/gated-fusion-mechanism
3)	Patrick W., Nils S., Ralf-Dieter B., Dieter K., Fatima I.L, Wojciech S., Tobias S., 25.05.2020, PTB-XL, a large publicly available electrocardiography dataset
 https://www.nature.com/articles/s41597-020-0495-6
4)	Sandra S., Krzysztof P., Damian L., 28.09.202, DOI: 10.3390/e23091121, ECG Signal Classification Using Deep Learning Techniques Based on the PTB-XL Dataset
https://.ncbi.nlm.nih.gov/articles/PMC8469424
5)	Aquib Irteza R., Valentina N,. Maria V., 30.08.2025, DOI: 10.2147/VHRM.S508620, Deep Learning-Based Detection of Arrhythmia Using ECG Signals - A Comprehensive Review
https://pmc.ncbi.nlm.nih.gov/articles/PMC12406999 
6)	ECG PTB XL Benchmarking – GitHub Repository
https://github.com/helme/ecg_ptbxl_benchmarking
7)	Helsinki-NLP/opus-mt-de-en – Hugging Face
https://huggingface.co/Helsinki-NLP/opus-mt-de-en
8)	Transformers – Hugging Face
https://huggingface.co/docs/transformers/index
9)	Tokenizing and Padding using Keras – Kaggle Notebook
https://www.kaggle.com/code/sajjadfc13/tokenizing-and-padding-using-keras/notebook
10)	Standard Scaler – Scikit Learn
https://scikit-learn.org/0.22/modules/generated/sklearn.preprocessing.StandardScaler.html
11)	PTB – XL ECG – 1D Convolution Neural Network
https://www.kaggle.com/code/jraska1/ptb-xl-ecg-1d-convolution-neural-network

12)	Conv1D – Tensor Flow
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
13)	ECG Classification | CNN + LSTM | Acc 98% - Kaggle Notebook
https://www.kaggle.com/code/behrouzmirabdi/ecg-classification-cnn-lstm-acc-98#Create-The-Model
14)	Model – Tensor Flow
https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
15)	Early Stopping – Tensor Flow
https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
16)	Metrics and scoring: quantifying the quality of predictions 
https://scikit-learn.org/stable/modules/model_evaluation.html
17)	Automatic ECG diagnosis – GitHub Repository
https://github.com/antonior92/automatic-ecg-diagnosis
18)	Naqcho Ali Mehdi, Aamir Ali Drigh, 08.03.2026, ECG Classification on PTB-XL: A Data-Centric Approach with Simplified CNN-VAE 
https://arxiv.org/pdf/2603.07558
19)	Mayo Clinic, Electrocardiogram (ECG or EKG)
https://www.mayoclinic.org/tests-procedures/ekg/about/pac-20384983

---

*This project is for research and educational purposes only. It is not a substitute for clinical diagnosis by a licensed physician.*
