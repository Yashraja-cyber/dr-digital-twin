"""Central configuration for the DR Digital Twin system.

All hyperparameters, paths, and clinical constants in one place.
"""
from __future__ import annotations

import os
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Create directories if they don't exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR, MODELS_DIR, ASSETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Image Configuration
# =============================================================================
IMAGE_SIZE = 384  # EfficientNet-B3 optimal input size
NUM_CLASSES = 5   # DR severity grades 0-4
CLASS_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
CLASS_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]

# =============================================================================
# Training Hyperparameters
# =============================================================================
MODEL_NAME = "efficientnet_b3"
BATCH_SIZE = 16          # Adjust based on GPU VRAM (16 for 8GB, 32 for 16GB+)
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 15
NUM_WORKERS = 4
MIXED_PRECISION = True
EARLY_STOPPING_PATIENCE = 5
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# Best model checkpoint path
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"

# =============================================================================
# Clinical Constants
# =============================================================================

# Annual base transition probabilities between DR grades
# Source: WESDR (Klein et al.), UKPDS, CURES (Rema et al.)
DR_TRANSITION_RATES = {
    (0, 1): 0.050,  # No DR → Mild (5% per year)
    (1, 2): 0.080,  # Mild → Moderate (8% per year)
    (2, 3): 0.120,  # Moderate → Severe (12% per year)
    (3, 4): 0.150,  # Severe → Proliferative (15% per year)
}

# Risk factor weights for composite risk score
# Evidence: UKPDS, DCCT/EDIC, WESDR
RISK_WEIGHTS = {
    "dr_grade": 0.30,
    "hba1c": 0.20,
    "diabetes_duration": 0.15,
    "blood_pressure": 0.10,
    "cgm_time_in_range": 0.10,
    "egfr": 0.05,
    "medication_adherence": 0.05,
    "lifestyle": 0.05,
}

# Risk score thresholds
RISK_THRESHOLDS = {
    "Low": (0, 20),
    "Moderate": (21, 40),
    "High": (41, 60),
    "Very High": (61, 80),
    "Critical": (81, 100),
}

# Screening interval recommendations (months)
SCREENING_INTERVALS = {
    "Low": 24,
    "Moderate": 12,
    "High": 6,
    "Very High": 3,
    "Critical": 0,  # Immediate referral
}

# Population-level DR grade distribution (India, approximate)
DR_GRADE_DISTRIBUTION = [0.45, 0.20, 0.18, 0.10, 0.07]

# HbA1c to estimated average glucose (eAG) conversion
# eAG (mg/dL) = 28.7 × HbA1c − 46.7 (Nathan et al., 2008)
HBAIC_TO_EAG_SLOPE = 28.7
HBAIC_TO_EAG_INTERCEPT = -46.7
