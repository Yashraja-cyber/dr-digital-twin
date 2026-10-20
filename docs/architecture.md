# System Architecture

## Overview

The DR Digital Twin system is organized into four major layers:

1. **Data Input Layer** — handles fundus image preprocessing, EHR ingestion, and wearable data parsing
2. **AI/ML Layer** — EfficientNet-B3 DR grading model with GradCAM explainability
3. **Digital Twin Engine** — risk scoring, progression simulation, and screening recommendations
4. **Presentation Layer** — Streamlit interactive dashboard with Plotly visualizations

## Component Details

### Image Preprocessing Pipeline

The preprocessing pipeline is optimized for variable-quality images from portable fundus cameras:

1. **Circle Crop** — Detects and removes black borders that vary across camera models
2. **Ben Graham's Method** — Subtracts local average color and rescales, originally developed for the Kaggle DR competition. Normalizes illumination and enhances lesion visibility
3. **CLAHE** — Contrast Limited Adaptive Histogram Equalization applied on the L channel in LAB color space. Enhances local contrast for detecting subtle microaneurysms
4. **Resize** — Standardizes to 384×384 pixels (EfficientNet-B3 optimal input)

### DR Grading Model

- **Backbone**: EfficientNet-B3 pretrained on ImageNet (1536 features)
- **Head**: Dropout → Linear(1536, 512) → BatchNorm → ReLU → Dropout → Linear(512, 5)
- **Training**: AdamW optimizer, cosine annealing LR, mixed precision, weighted sampling
- **Evaluation**: Quadratic Weighted Kappa (QWK), per-class F1

### Risk Calculator

Composite score (0-100) with evidence-backed weights:

| Factor | Weight | Normalization Range | Source |
|---|---|---|---|
| DR Grade | 30% | 0-4 grades | WESDR |
| HbA1c | 20% | 5.5-12% | DCCT/EDIC |
| Diabetes Duration | 15% | 0-25 years | UKPDS |
| Blood Pressure | 10% | 110-180 mmHg | UKPDS |
| CGM Time-in-Range | 10% | 0-100% (inverted) | ADA Consensus |
| eGFR | 5% | 15-120 mL/min (inverted) | WESDR |
| Medication Adherence | 5% | 30-100% (inverted) | Literature |
| Lifestyle (BMI+Smoking) | 5% | Composite | UKPDS |

### Progression Model

5-state Markov Chain with risk-adjusted annual transition probabilities:

| Transition | Base Rate | Source |
|---|---|---|
| No DR → Mild | 5%/year | WESDR |
| Mild → Moderate | 8%/year | WESDR |
| Moderate → Severe | 12%/year | UKPDS |
| Severe → Proliferative | 15%/year | UKPDS |

- Risk multiplier: score 0 → 0.3× (protective), score 50 → 1.0× (baseline), score 100 → 2.0× (accelerated)
- Monte Carlo: 1000 simulations × 5 years
- No backward transitions (DR does not spontaneously reverse)
- State 4 (Proliferative) is absorbing

### Screening Recommender

Risk-adaptive intervals based on ICO/AAO guidelines:

| Risk Score | Interval | Urgency | Referral |
|---|---|---|---|
| 0-20 | 24 months | Routine | Primary Care |
| 21-40 | 12 months | Routine | Ophthalmologist |
| 41-60 | 6 months | Priority | Ophthalmologist |
| 61-80 | 3 months | Urgent | Retina Specialist |
| 81-100 | Immediate | Emergency | Retina Specialist |

Override: If DR grade ≥ 3, always refer to specialist regardless of risk score.
