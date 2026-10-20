# 🏥 DR Digital Twin — Diabetic Retinopathy Progression Prediction System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **An AI-powered Healthcare Digital Twin that combines fundus image analysis, electronic health records, and wearable sensor data to predict diabetic retinopathy progression and deliver personalized screening recommendations.**

Built for the [**CRP Digital Twin Challenge 2026**](https://unstop.com/hackathons/crp-digital-twin-challenge-2026-happiest-health-1757873) by Happiest Health.

---

## 🎯 Problem Statement

India has over **80 million diabetic adults** — the second highest globally. Diabetic Retinopathy (DR) affects ~18% of this population and is a **leading cause of preventable blindness**. Early screening can prevent up to 90% of vision loss, but India has barely **1 ophthalmologist per 100,000 rural population**, making mass manual screening infeasible.

Existing AI solutions function as black boxes, lack clinical validation rigor, and fail with variable image quality from portable fundus cameras in field conditions. **What's missing is a longitudinal, patient-specific model** — a Digital Twin — that continuously integrates health records, wearable data, and retinal imaging to simulate disease progression and personalize care.

## 💡 Our Solution

A **Diabetic Retinopathy Progression Digital Twin** — a dynamic virtual replica of each patient that:

1. **Grades fundus images** using an EfficientNet-B3 CNN trained on the APTOS 2019 dataset (5-class ICDR scale)
2. **Explains predictions** with GradCAM heatmaps showing which retinal regions drive each diagnosis
3. **Integrates EHR data** (HbA1c, blood pressure, eGFR, medications, comorbidities)
4. **Processes wearable data** (continuous glucose monitor streams, heart rate, activity)
5. **Computes composite risk** using evidence-weighted scoring from UKPDS, DCCT, and WESDR clinical trials
6. **Simulates progression** via Markov Chain Monte Carlo over a 5-year horizon
7. **Recommends screening intervals** personalized to each patient's risk profile

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA INPUT LAYER                        │
│  📷 Fundus Image    📋 EHR Records    ⌚ Wearable/CGM       │
└────────┬──────────────────┬──────────────────┬──────────────┘
         │                  │                  │
    ┌────▼────┐      ┌─────▼─────┐     ┌─────▼──────┐
    │ Preproc │      │  Patient  │     │    CGM     │
    │ CLAHE + │      │  Profile  │     │  Summary   │
    │BenGraham│      │  Builder  │     │ Calculator │
    └────┬────┘      └─────┬─────┘     └─────┬──────┘
         │                 │                  │
    ┌────▼────┐      ┌─────▼──────────────────▼──────┐
    │EfficNet │      │                               │
    │ B3 CNN  │      │     DIGITAL TWIN ENGINE       │
    │ Grading │─────►│  ┌─────────────────────────┐  │
    └────┬────┘      │  │  Risk Calculator (0-100) │  │
         │           │  │  8 evidence-weighted      │  │
    ┌────▼────┐      │  │  clinical factors         │  │
    │ GradCAM │      │  └────────────┬────────────┘  │
    │Heatmaps │      │  ┌────────────▼────────────┐  │
    └─────────┘      │  │  Markov Progression     │  │
                     │  │  Monte Carlo Simulator  │  │
                     │  └────────────┬────────────┘  │
                     │  ┌────────────▼────────────┐  │
                     │  │  Screening Recommender  │  │
                     │  │  Personalized intervals │  │
                     │  └─────────────────────────┘  │
                     └──────────────┬─────────────────┘
                                   │
                     ┌─────────────▼──────────────┐
                     │   STREAMLIT DASHBOARD       │
                     │  ┌────────┐ ┌────────────┐  │
                     │  │Fundus +│ │Risk Gauge  │  │
                     │  │GradCAM │ │+ Breakdown │  │
                     │  ├────────┤ ├────────────┤  │
                     │  │CGM Plot│ │Progression │  │
                     │  │+Stats  │ │Forecast    │  │
                     │  └────────┘ └────────────┘  │
                     └─────────────────────────────┘
```

## 📂 Project Structure

```
dr-digital-twin/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── .gitignore
│
├── src/
│   ├── config.py                          # Central configuration
│   ├── data/
│   │   ├── preprocessing.py               # CLAHE, Ben Graham, circle crop
│   │   ├── augmentation.py                # Albumentations training pipeline
│   │   ├── dataset.py                     # PyTorch Dataset (APTOS 2019)
│   │   └── synthetic_ehr.py               # Synthetic EHR + CGM generator
│   ├── models/
│   │   ├── dr_grader.py                   # EfficientNet-B3 DR grading CNN
│   │   └── explainability.py              # GradCAM visual explanations
│   ├── digital_twin/
│   │   ├── twin_engine.py                 # Central orchestrator
│   │   ├── risk_calculator.py             # Multi-factor risk scoring
│   │   ├── progression_model.py           # Markov Chain Monte Carlo
│   │   └── screening_recommender.py       # Personalized screening
│   └── train.py                           # Model training script
│
├── dashboard/
│   └── app.py                             # Streamlit interactive dashboard
│
├── docs/
│   ├── architecture.md                    # Detailed architecture docs
│   └── clinical_rationale.md              # Clinical evidence & references
│
└── data/                                  # (gitignored)
    ├── raw/                               # Downloaded datasets
    ├── processed/                         # Preprocessed images
    └── synthetic/                         # Generated patient data
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU (recommended for training; dashboard runs on CPU)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/dr-digital-twin.git
cd dr-digital-twin

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard (Demo Mode)

The dashboard works **without a trained model** using demo mode with synthetic data:

```bash
streamlit run dashboard/app.py
```

This launches the interactive dashboard where you can:
- Adjust patient parameters (age, HbA1c, duration, BP, etc.)
- View real-time risk scoring
- See 5-year progression forecasts
- Get personalized screening recommendations
- Upload fundus images for AI grading (when model is trained)

### Train the Model (Optional)

1. Download the [APTOS 2019 dataset](https://www.kaggle.com/c/aptos2019-blindness-detection) from Kaggle
2. Extract to `data/raw/aptos2019/`
3. Run training:

```bash
python src/train.py --data_dir data/raw/aptos2019 --epochs 15 --batch_size 16
```

Training takes ~30 minutes on a single GPU. The best model is saved to `models/best_model.pth`.

---

## 📊 Datasets Used

| Dataset | Purpose | Size | Source |
|---|---|---|---|
| **APTOS 2019** | DR grading model training | 3,662 images | [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection) |
| **IDRiD** | Lesion segmentation + validation | 516 images | [IEEE Dataport](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) |
| **DRIVE** | Retinal vessel extraction | 40 images | [Grand Challenge](https://drive.grand-challenge.org/) |
| **Messidor-2** | Cross-dataset validation | 1,748 images | [ADCIS](https://www.adcis.net/en/third-party/messidor2/) |

## 🔬 Clinical Evidence

Risk factor weights and transition probabilities are derived from:
- **WESDR** — Wisconsin Epidemiologic Study of Diabetic Retinopathy (Klein et al.)
- **UKPDS** — UK Prospective Diabetes Study
- **DCCT/EDIC** — Diabetes Control and Complications Trial
- **CURES** — Chennai Urban Rural Epidemiology Study (Rema et al.)
- **SN-DREAMS** — Sankara Nethralaya Diabetic Retinopathy Epidemiology and Molecular Genetics Study
- **ICMR-INDIAB** — Indian Council of Medical Research–India Diabetes Study

## 🔑 Key Technical Features

| Feature | Implementation |
|---|---|
| **Explainable AI** | GradCAM heatmaps showing lesion-level attention regions |
| **Ordinal Classification** | 5-class ICDR scale with class-weighted training |
| **Quality Robustness** | Ben Graham preprocessing + CLAHE for portable camera images |
| **Temporal Modeling** | Risk-adjusted Markov Chain with Monte Carlo sampling |
| **Evidence-Based Risk** | 8 clinical factors with literature-backed weights |
| **Personalized Screening** | Risk-adaptive intervals from 24 months to immediate referral |
| **Interactive Dashboard** | Real-time Streamlit UI with Plotly visualizations |

## ⚠️ Limitations & Ethical Considerations

- **Synthetic data**: EHR and wearable data are synthetically generated for this PoC. The architecture is designed for real data integration.
- **Not clinically validated**: This system is a technology demonstration and is **not intended for clinical decision-making** without proper clinical trials and regulatory approval.
- **Cross-sectional training**: The AI model is trained on single-timepoint images. Real longitudinal data would improve progression predictions.
- **Bias considerations**: The APTOS dataset is from the Asia-Pacific region. Further validation is needed across diverse populations.

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

*Built with ❤️ for the CRP Digital Twin Challenge 2026 by Happiest Health*
