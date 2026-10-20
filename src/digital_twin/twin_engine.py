"""Central Digital Twin orchestrator.

The DigitalTwin class ties together all components — EHR data,
wearable data, fundus image analysis, risk scoring, progression
modeling, and screening recommendations — into a single cohesive
patient-level digital replica.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from src.config import CLASS_NAMES
from src.digital_twin.progression_model import DRProgressionModel
from src.digital_twin.risk_calculator import RiskCalculator
from src.digital_twin.screening_recommender import ScreeningRecommender


class DigitalTwin:
    """A patient-specific digital twin for DR progression.

    Orchestrates the full analysis pipeline:
    1. Ingest EHR data (demographics, labs, medications)
    2. Ingest wearable data (CGM glucose stream)
    3. Set fundus image analysis results (grade + GradCAM)
    4. Compute composite risk score
    5. Simulate future progression (Monte Carlo)
    6. Generate personalized screening recommendations

    Args:
        patient_id: Unique patient identifier.
    """

    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.patient_data: dict = {}
        self.wearable_data: pd.DataFrame | None = None
        self.wearable_summary: dict | None = None
        self.fundus_grade: int | None = None
        self.fundus_confidence: float | None = None
        self.fundus_probabilities: dict | None = None
        self.gradcam_heatmap: np.ndarray | None = None
        self.risk_assessment: dict | None = None
        self.progression_forecast: dict | None = None
        self.screening_recommendation: dict | None = None
        self.created_at = datetime.now()
        self.last_updated = datetime.now()

        # Internal components
        self._risk_calc = RiskCalculator()
        self._progression_model = DRProgressionModel()
        self._screening_rec = ScreeningRecommender()

    def ingest_ehr(self, ehr_data: dict) -> None:
        """Load patient EHR (demographics, labs, medications).

        Args:
            ehr_data: Dictionary from synthetic_ehr.generate_patient()
                      or real EHR data with matching keys.
        """
        self.patient_data = ehr_data.copy()
        if "dr_grade" in ehr_data and self.fundus_grade is None:
            self.fundus_grade = ehr_data["dr_grade"]
        self.last_updated = datetime.now()

    def ingest_wearable(self, cgm_df: pd.DataFrame) -> None:
        """Load wearable/CGM data stream and compute summary.

        Args:
            cgm_df: DataFrame from synthetic_ehr.generate_cgm_stream().
        """
        from src.data.synthetic_ehr import generate_wearable_summary

        self.wearable_data = cgm_df
        self.wearable_summary = generate_wearable_summary(cgm_df)
        self.last_updated = datetime.now()

    def set_fundus_result(
        self,
        grade: int,
        confidence: float,
        probabilities: dict | None = None,
        gradcam: np.ndarray | None = None,
    ) -> None:
        """Store fundus image analysis results.

        Args:
            grade: Predicted DR grade (0-4).
            confidence: Model confidence (0-1).
            probabilities: Per-class probabilities.
            gradcam: GradCAM heatmap array (H, W).
        """
        self.fundus_grade = grade
        self.fundus_confidence = confidence
        self.fundus_probabilities = probabilities
        self.gradcam_heatmap = gradcam
        # Sync with patient data
        self.patient_data["dr_grade"] = grade
        self.patient_data["dr_grade_label"] = CLASS_NAMES[grade]
        self.last_updated = datetime.now()

    def compute_risk(self) -> dict:
        """Compute the composite risk score.

        Returns:
            Risk assessment dictionary with total_score, category,
            factor_breakdown, and contributing_factors.
        """
        if not self.patient_data:
            raise ValueError("No EHR data ingested. Call ingest_ehr() first.")

        # Ensure DR grade is set
        if "dr_grade" not in self.patient_data and self.fundus_grade is not None:
            self.patient_data["dr_grade"] = self.fundus_grade

        self.risk_assessment = self._risk_calc.compute_risk_score(
            self.patient_data,
            self.wearable_summary,
        )
        self.last_updated = datetime.now()
        return self.risk_assessment

    def simulate_progression(self, years: int = 5) -> dict:
        """Run Monte Carlo progression simulation.

        Args:
            years: Number of years to simulate.

        Returns:
            Progression forecast dictionary.
        """
        if self.risk_assessment is None:
            self.compute_risk()

        current_grade = self.patient_data.get("dr_grade", self.fundus_grade or 0)
        risk_score = self.risk_assessment["total_score"]

        self.progression_forecast = self._progression_model.simulate_progression(
            current_grade=current_grade,
            risk_score=risk_score,
            years=years,
        )
        self.last_updated = datetime.now()
        return self.progression_forecast

    def get_screening_recommendation(self) -> dict:
        """Generate personalized screening recommendation.

        Returns:
            Screening recommendation dictionary.
        """
        if self.risk_assessment is None:
            self.compute_risk()
        if self.progression_forecast is None:
            self.simulate_progression()

        current_grade = self.patient_data.get("dr_grade", self.fundus_grade or 0)
        risk_score = self.risk_assessment["total_score"]

        self.screening_recommendation = self._screening_rec.recommend(
            risk_score=risk_score,
            current_grade=current_grade,
            progression_data=self.progression_forecast,
        )
        self.last_updated = datetime.now()
        return self.screening_recommendation

    def run_full_analysis(self) -> dict:
        """Run the complete analysis pipeline.

        Executes risk scoring, progression simulation, and screening
        recommendation in sequence.

        Returns:
            Complete analysis results.
        """
        risk = self.compute_risk()
        progression = self.simulate_progression()
        screening = self.get_screening_recommendation()

        return {
            "patient_id": self.patient_id,
            "risk_assessment": risk,
            "progression_forecast": {
                k: v for k, v in progression.items() if k != "simulations"
            },
            "screening_recommendation": screening,
        }

    def to_dashboard_payload(self) -> dict:
        """Package all twin state for dashboard rendering.

        Returns:
            Dictionary with all data needed by the Streamlit dashboard.
        """
        # Run analysis if not yet done
        if self.risk_assessment is None:
            self.run_full_analysis()

        # Build progression data without the raw simulations array
        progression_clean = None
        if self.progression_forecast:
            progression_clean = {
                k: v for k, v in self.progression_forecast.items()
                if k != "simulations"
            }

        return {
            "patient_id": self.patient_id,
            "patient_data": self.patient_data,
            "wearable_summary": self.wearable_summary,
            "fundus_grade": self.fundus_grade,
            "fundus_confidence": self.fundus_confidence,
            "fundus_probabilities": self.fundus_probabilities,
            "risk_assessment": self.risk_assessment,
            "progression_forecast": progression_clean,
            "screening_recommendation": self.screening_recommendation,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
        }

    def get_summary_text(self) -> str:
        """Generate a human-readable clinical summary.

        Returns:
            Multi-line clinical summary string.
        """
        if self.risk_assessment is None:
            self.run_full_analysis()

        p = self.patient_data
        risk = self.risk_assessment
        prog = self.progression_forecast or {}
        screen = self.screening_recommendation or {}

        grade = self.fundus_grade if self.fundus_grade is not None else 0
        grade_label = CLASS_NAMES[grade]
        confidence = self.fundus_confidence or 0.0

        lines = [
            f"═══ DIGITAL TWIN CLINICAL SUMMARY ═══",
            f"Patient: {self.patient_id}",
            f"Generated: {self.last_updated.strftime('%Y-%m-%d %H:%M')}",
            f"",
            f"── PATIENT PROFILE ──",
            f"Age: {p.get('age', 'N/A')} | Gender: {p.get('gender', 'N/A')}",
            f"Diabetes Type: {p.get('diabetes_type', 'Type 2')} | Duration: {p.get('diabetes_duration_years', 'N/A')} years",
            f"HbA1c: {p.get('hba1c', 'N/A')}% | Fasting Glucose: {p.get('fasting_glucose_mg_dl', 'N/A')} mg/dL",
            f"BP: {p.get('systolic_bp', 'N/A')}/{p.get('diastolic_bp', 'N/A')} mmHg | BMI: {p.get('bmi', 'N/A')}",
            f"eGFR: {p.get('egfr_ml_min', 'N/A')} mL/min",
            f"",
            f"── FUNDUS ANALYSIS ──",
            f"DR Grade: {grade_label} (Grade {grade}/4)",
            f"Model Confidence: {confidence:.1%}",
            f"",
            f"── RISK ASSESSMENT ──",
            f"Composite Risk Score: {risk['total_score']:.1f}/100",
            f"Risk Category: {risk['category']}",
            f"",
            f"── 5-YEAR PROGRESSION FORECAST ──",
            f"Probability of progression: {prog.get('probability_of_progression', 0):.1%}",
            f"Worst-case grade (95th %ile): {prog.get('worst_case_grade', 'N/A')}",
            f"Median time to progression: {prog.get('median_time_to_progression_years', 'N/A')} years",
            f"",
            f"── SCREENING RECOMMENDATION ──",
            f"Next screening: {screen.get('screening_interval_months', 'N/A')} months",
            f"Urgency: {screen.get('urgency', 'N/A')}",
            f"Referral: {screen.get('referral_type', 'N/A')}",
            f"",
            f"⚠️ DISCLAIMER: This is a PoC digital twin for research purposes.",
            f"Not intended for clinical decision-making without physician review.",
        ]
        return "\n".join(lines)
