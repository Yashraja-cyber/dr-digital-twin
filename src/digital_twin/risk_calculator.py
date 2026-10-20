"""Multi-factor risk scoring for diabetic retinopathy progression.

Factor weights are derived from landmark clinical studies:
- UKPDS (UK Prospective Diabetes Study)
- DCCT (Diabetes Control and Complications Trial)
- WESDR (Wisconsin Epidemiologic Study of Diabetic Retinopathy)

The risk score integrates EHR data, wearable device metrics, and current
disease grade into a single 0-100 composite score used downstream for
progression modelling and screening interval determination.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clinical reference ranges used for normalization
# ---------------------------------------------------------------------------
_CLINICAL_RANGES: dict[str, dict[str, float]] = {
    "hba1c": {"min": 4.0, "max": 14.0},                # % (NGSP)
    "diabetes_duration_years": {"min": 0.0, "max": 40.0},
    "systolic_bp": {"min": 90.0, "max": 200.0},         # mmHg
    "egfr": {"min": 15.0, "max": 120.0},                # mL/min/1.73m²
    "bmi": {"min": 18.5, "max": 45.0},
    "medication_adherence": {"min": 0.0, "max": 100.0},  # %
    "cgm_time_in_range": {"min": 0.0, "max": 100.0},     # %
}


class RiskCalculator:
    """Multi-factor risk scoring for DR progression on a 0-100 scale.

    Factor weights (summing to 1.0) based on UKPDS / DCCT / WESDR evidence:

    =============================  ======
    Factor                         Weight
    =============================  ======
    current_dr_grade                0.30
    hba1c                           0.20
    diabetes_duration_years         0.15
    systolic_bp                     0.10
    cgm_time_in_range               0.10
    egfr                            0.05
    medication_adherence            0.05
    lifestyle (bmi + smoking)       0.05
    =============================  ======

    When CGM data is unavailable the 10 % CGM weight is redistributed
    proportionally across the remaining factors.
    """

    # Base factor weights (order mirrors the docstring table)
    _BASE_WEIGHTS: dict[str, float] = {
        "current_dr_grade": 0.30,
        "hba1c": 0.20,
        "diabetes_duration_years": 0.15,
        "systolic_bp": 0.10,
        "cgm_time_in_range": 0.10,
        "egfr": 0.05,
        "medication_adherence": 0.05,
        "lifestyle": 0.05,
    }

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def compute_risk_score(
        self,
        patient_data: dict[str, Any],
        wearable_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute the composite DR risk score for a patient.

        Parameters
        ----------
        patient_data:
            Dictionary with keys such as ``current_dr_grade``, ``hba1c``,
            ``diabetes_duration_years``, ``systolic_bp``, ``egfr``,
            ``medication_adherence``, ``bmi``, ``smoking``.
        wearable_summary:
            Optional summary from CGM processing containing at least
            ``time_in_range`` (0-100 percentage).

        Returns
        -------
        dict
            ``total_score``  – composite risk score (0-100).
            ``category``     – human-readable risk category.
            ``factor_breakdown`` – per-factor weighted contributions.
            ``contributing_factors`` – ordered list of top risk drivers.
        """
        breakdown = self.get_factor_breakdown(patient_data, wearable_summary)

        total_score: float = sum(
            entry["weighted_score"] for entry in breakdown.values()
        )
        # Clamp to [0, 100]
        total_score = max(0.0, min(100.0, total_score))

        category = self.get_risk_category(total_score)

        # Rank contributing factors by weighted score descending
        contributing_factors = sorted(
            breakdown.keys(),
            key=lambda k: breakdown[k]["weighted_score"],
            reverse=True,
        )

        logger.info(
            "Risk score computed: %.1f (%s) for patient data with %d factors",
            total_score,
            category,
            len(breakdown),
        )

        return {
            "total_score": round(total_score, 2),
            "category": category,
            "factor_breakdown": breakdown,
            "contributing_factors": contributing_factors,
        }

    # --------------------------------------------------------------------- #
    # Factor breakdown
    # --------------------------------------------------------------------- #

    def get_factor_breakdown(
        self,
        patient_data: dict[str, Any],
        wearable_summary: dict[str, Any] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Return each factor's raw score (0-1) and weighted contribution.

        Parameters
        ----------
        patient_data:
            EHR-derived patient record.
        wearable_summary:
            Optional CGM summary.

        Returns
        -------
        dict
            Mapping of ``factor_name`` → ``{"raw_score": float,
            "weight": float, "weighted_score": float}``.
        """
        has_cgm = (
            wearable_summary is not None
            and "time_in_range_pct" in wearable_summary
        )

        weights = dict(self._BASE_WEIGHTS)
        if not has_cgm:
            # Remove CGM weight and redistribute proportionally
            cgm_weight = weights.pop("cgm_time_in_range")
            remaining_total = sum(weights.values())
            for key in weights:
                weights[key] += cgm_weight * (weights[key] / remaining_total)

        breakdown: dict[str, dict[str, float]] = {}

        # --- current DR grade (higher grade → higher risk) ---------------
        raw = self._score_dr_grade(patient_data.get("current_dr_grade", 0))
        breakdown["current_dr_grade"] = self._entry(raw, weights["current_dr_grade"])

        # --- HbA1c (higher → worse) -------------------------------------
        raw = self._normalize_factor(
            patient_data.get("hba1c", 5.7),
            _CLINICAL_RANGES["hba1c"]["min"],
            _CLINICAL_RANGES["hba1c"]["max"],
        )
        breakdown["hba1c"] = self._entry(raw, weights["hba1c"])

        # --- Diabetes duration (longer → worse) --------------------------
        raw = self._normalize_factor(
            patient_data.get("diabetes_duration_years", 0),
            _CLINICAL_RANGES["diabetes_duration_years"]["min"],
            _CLINICAL_RANGES["diabetes_duration_years"]["max"],
        )
        breakdown["diabetes_duration_years"] = self._entry(
            raw, weights["diabetes_duration_years"]
        )

        # --- Systolic BP (higher → worse) --------------------------------
        raw = self._normalize_factor(
            patient_data.get("systolic_bp", 120),
            _CLINICAL_RANGES["systolic_bp"]["min"],
            _CLINICAL_RANGES["systolic_bp"]["max"],
        )
        breakdown["systolic_bp"] = self._entry(raw, weights["systolic_bp"])

        # --- CGM time-in-range (higher → *better*, so invert) ------------
        if has_cgm:
            raw = self._normalize_factor(
                wearable_summary["time_in_range_pct"],  # type: ignore[index]
                _CLINICAL_RANGES["cgm_time_in_range"]["min"],
                _CLINICAL_RANGES["cgm_time_in_range"]["max"],
                invert=True,
            )
            breakdown["cgm_time_in_range"] = self._entry(
                raw, weights["cgm_time_in_range"]
            )

        # --- eGFR (lower → worse, so invert) -----------------------------
        raw = self._normalize_factor(
            patient_data.get("egfr", 90),
            _CLINICAL_RANGES["egfr"]["min"],
            _CLINICAL_RANGES["egfr"]["max"],
            invert=True,
        )
        breakdown["egfr"] = self._entry(raw, weights["egfr"])

        # --- Medication adherence (higher → *better*, so invert) ---------
        raw = self._normalize_factor(
            patient_data.get("medication_adherence", 80),
            _CLINICAL_RANGES["medication_adherence"]["min"],
            _CLINICAL_RANGES["medication_adherence"]["max"],
            invert=True,
        )
        breakdown["medication_adherence"] = self._entry(
            raw, weights["medication_adherence"]
        )

        # --- Lifestyle composite (BMI + smoking) -------------------------
        raw = self._score_lifestyle(
            bmi=patient_data.get("bmi", 25.0),
            smoking=patient_data.get("smoking", False),
        )
        breakdown["lifestyle"] = self._entry(raw, weights["lifestyle"])

        return breakdown

    # --------------------------------------------------------------------- #
    # Risk category mapping
    # --------------------------------------------------------------------- #

    @staticmethod
    def get_risk_category(score: float) -> str:
        """Map a 0-100 risk score to a human-readable category.

        Categories
        ----------
        - **Low**: 0-20
        - **Moderate**: 21-40
        - **High**: 41-60
        - **Very High**: 61-80
        - **Critical**: 81-100
        """
        if score <= 20:
            return "Low"
        if score <= 40:
            return "Moderate"
        if score <= 60:
            return "High"
        if score <= 80:
            return "Very High"
        return "Critical"

    # --------------------------------------------------------------------- #
    # Normalization helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _normalize_factor(
        value: float,
        min_val: float,
        max_val: float,
        invert: bool = False,
    ) -> float:
        """Linearly normalise *value* to the 0-1 range.

        Parameters
        ----------
        value:
            Raw clinical measurement.
        min_val:
            Lower bound of the expected clinical range.
        max_val:
            Upper bound of the expected clinical range.
        invert:
            If ``True`` the scale is reversed so that *higher* raw values
            map to *lower* risk (e.g. eGFR, medication adherence).

        Returns
        -------
        float
            Normalised score clamped to [0, 1].
        """
        if max_val == min_val:
            return 0.0
        normalised = (float(value) - min_val) / (max_val - min_val)
        normalised = max(0.0, min(1.0, normalised))
        if invert:
            normalised = 1.0 - normalised
        return normalised

    # --------------------------------------------------------------------- #
    # Internal scoring helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _score_dr_grade(grade: int | None) -> float:
        """Convert a 0-4 DR grade into a 0-1 risk factor.

        Mapping is non-linear to reflect the accelerating clinical risk
        at higher grades:
        0 → 0.0, 1 → 0.20, 2 → 0.45, 3 → 0.75, 4 → 1.0
        """
        grade_map: dict[int, float] = {
            0: 0.00,
            1: 0.20,
            2: 0.45,
            3: 0.75,
            4: 1.00,
        }
        if grade is None:
            return 0.0
        return grade_map.get(int(grade), 1.0)

    @staticmethod
    def _score_lifestyle(bmi: float, smoking: bool) -> float:
        """Compute a combined lifestyle risk factor (0-1).

        BMI contribution (70 % of lifestyle weight):
        - 18.5-24.9 → 0.0 (normal)
        - 25-29.9   → 0.4 (overweight)
        - 30-34.9   → 0.7 (obese I)
        - ≥35       → 1.0 (obese II+)

        Smoking contribution (30 % of lifestyle weight):
        - True → 1.0
        - False → 0.0
        """
        if bmi < 25.0:
            bmi_risk = 0.0
        elif bmi < 30.0:
            bmi_risk = 0.4
        elif bmi < 35.0:
            bmi_risk = 0.7
        else:
            bmi_risk = 1.0

        smoking_risk = 1.0 if smoking else 0.0
        return 0.7 * bmi_risk + 0.3 * smoking_risk

    @staticmethod
    def _entry(raw_score: float, weight: float) -> dict[str, float]:
        """Build a factor-breakdown entry dictionary."""
        return {
            "raw_score": round(raw_score, 4),
            "weight": round(weight, 4),
            "weighted_score": round(raw_score * weight * 100, 4),
        }
