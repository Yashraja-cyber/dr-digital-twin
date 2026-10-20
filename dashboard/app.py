from __future__ import annotations

import logging
import math
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup — allow running from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Internal imports (digital-twin back-end)
# ---------------------------------------------------------------------------
from src.data.synthetic_ehr import (  # noqa: E402
    generate_cgm_stream,
    generate_patient,
    generate_wearable_summary,
)
from src.digital_twin.progression_model import DRProgressionModel  # noqa: E402
from src.digital_twin.risk_calculator import RiskCalculator  # noqa: E402
from src.digital_twin.screening_recommender import ScreeningRecommender  # noqa: E402
from src.digital_twin.twin_engine import DigitalTwin  # noqa: E402

# ---------------------------------------------------------------------------
# Optional deep-learning imports (model + explainability)
# ---------------------------------------------------------------------------
try:
    from src.models.dr_grader import DRGrader, load_model, predict_image  # noqa: E402
    from src.models.explainability import GradCAMExplainer  # noqa: E402

    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DR_GRADE_LABELS: Dict[int, str] = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

DR_GRADE_COLORS: Dict[int, str] = {
    0: "#2ecc71",
    1: "#f1c40f",
    2: "#e67e22",
    3: "#e74c3c",
    4: "#8b0000",
}

CHECKPOINT_PATH = os.path.join(_PROJECT_ROOT, "models", "best_model.pth")

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DR Digital Twin",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark medical theme
# ---------------------------------------------------------------------------
_CUSTOM_CSS = """
<style>
/* ---- global overrides ---- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #0d1117 0%, #161b22 100%);
}
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
h1, h2, h3, h4 {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* ---- metric cards ---- */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 2px 12px rgba(0,0,0,.35);
}
[data-testid="stMetricLabel"] {
    color: #8b949e !important;
    font-weight: 600;
}
[data-testid="stMetricValue"] {
    color: #e6edf3 !important;
    font-weight: 700;
}

/* ---- DR grade badges ---- */
.dr-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: .3px;
    margin-top: 4px;
}

/* ---- panel cards ---- */
.panel-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 4px 20px rgba(0,0,0,.30);
    margin-bottom: 12px;
}
.panel-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #58a6ff;
    margin-bottom: 10px;
}

/* ---- recommendation box ---- */
.rec-box {
    background: #0d1117;
    border-left: 4px solid #58a6ff;
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 12px;
}
.rec-box h4 { margin: 0 0 6px 0; color: #79c0ff; }
.rec-item { margin: 3px 0; color: #c9d1d9; }

/* ---- disclaimer ---- */
.disclaimer {
    color: #8b949e;
    font-size: .82rem;
    border-top: 1px solid #21262d;
    padding-top: 12px;
    margin-top: 24px;
}
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


# ===================================================================
# Helper utilities
# ===================================================================


def _estimate_dr_grade_from_ehr(hba1c: float, duration: float) -> int:
    """Heuristic DR grade estimate when no fundus image / model is available.

    Combines HbA1c and diabetes duration with a small stochastic component
    so that repeated runs produce slightly varying results, mimicking the
    uncertainty inherent in clinical grading.
    """
    score = (hba1c - 6.0) * 1.5 + (duration - 3) * 0.35 + random.gauss(0, 0.6)
    if score < 1.5:
        return 0
    if score < 3.5:
        return 1
    if score < 5.5:
        return 2
    if score < 7.5:
        return 3
    return 4


def _grade_badge_html(grade: int) -> str:
    """Return an HTML badge for a given DR grade."""
    color = DR_GRADE_COLORS.get(grade, "#ccc")
    label = DR_GRADE_LABELS.get(grade, "Unknown")
    text_color = "#fff" if grade >= 2 else "#1a1a1a"
    return (
        f'<span class="dr-badge" style="background:{color};color:{text_color};">'
        f"Grade {grade} — {label}</span>"
    )


def _build_confidence_bar(confidence: float) -> None:
    """Render a horizontal confidence bar via ``st.progress``."""
    pct = int(round(confidence * 100))
    st.caption(f"Model confidence: **{pct}%**")
    st.progress(min(confidence, 1.0))


# ===================================================================
# Chart builders (Plotly, dark template)
# ===================================================================

_PLOTLY_LAYOUT_DEFAULTS: Dict[str, Any] = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=40, b=40),
    font=dict(family="Segoe UI, sans-serif", size=12),
)


def _make_gauge_chart(risk_score: float) -> go.Figure:
    """Create a Plotly gauge (0-100) with green/yellow/orange/red zones."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=risk_score,
            number={"suffix": "%", "font": {"size": 42, "color": "#e6edf3"}},
            title={"text": "Overall DR Risk Score", "font": {"size": 16, "color": "#8b949e"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#484f58"},
                "bar": {"color": "#58a6ff"},
                "bgcolor": "#21262d",
                "steps": [
                    {"range": [0, 25], "color": "#238636"},
                    {"range": [25, 50], "color": "#d29922"},
                    {"range": [50, 75], "color": "#da3633"},
                    {"range": [75, 100], "color": "#8b0000"},
                ],
                "threshold": {
                    "line": {"color": "#f0f6fc", "width": 3},
                    "thickness": 0.8,
                    "value": risk_score,
                },
            },
        )
    )
    fig.update_layout(**_PLOTLY_LAYOUT_DEFAULTS, height=280)
    return fig


def _make_radar_chart(factor_scores: Dict[str, float]) -> go.Figure:
    """Radar/spider chart of individual risk-factor contributions."""
    categories = list(factor_scores.keys())
    values = list(factor_scores.values())
    # Close the polygon
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(88,166,255,0.18)",
            line=dict(color="#58a6ff", width=2),
            marker=dict(size=6, color="#79c0ff"),
            name="Risk Factors",
        )
    )
    fig.update_layout(
        **_PLOTLY_LAYOUT_DEFAULTS,
        height=310,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="#30363d",
                tickfont=dict(size=10, color="#8b949e"),
            ),
            angularaxis=dict(gridcolor="#30363d", tickfont=dict(size=11, color="#c9d1d9")),
        ),
        showlegend=False,
    )
    return fig


def _make_cgm_chart(timestamps: List[Any], glucose_values: List[float]) -> go.Figure:
    """7-day CGM line chart with normal/hypo/hyper colour bands."""
    fig = go.Figure()

    # Shaded zones
    x_range = [timestamps[0], timestamps[-1]]
    # Hypo zone (< 70)
    fig.add_shape(
        type="rect",
        x0=x_range[0], x1=x_range[1], y0=30, y1=70,
        fillcolor="rgba(218,54,51,0.12)", line_width=0, layer="below",
    )
    # Normal zone (70-180)
    fig.add_shape(
        type="rect",
        x0=x_range[0], x1=x_range[1], y0=70, y1=180,
        fillcolor="rgba(35,134,54,0.12)", line_width=0, layer="below",
    )
    # Hyper zone (> 250)
    fig.add_shape(
        type="rect",
        x0=x_range[0], x1=x_range[1], y0=250, y1=420,
        fillcolor="rgba(218,54,51,0.12)", line_width=0, layer="below",
    )

    # Reference lines
    for yval, label, color in [
        (70, "Hypo", "#da3633"),
        (180, "Target", "#238636"),
        (250, "Hyper", "#da3633"),
    ]:
        fig.add_hline(y=yval, line_dash="dot", line_color=color, opacity=0.5,
                      annotation_text=label, annotation_position="top left",
                      annotation_font_color=color)

    # CGM trace
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=glucose_values,
            mode="lines",
            line=dict(color="#58a6ff", width=1.6),
            name="CGM Glucose",
        )
    )

    fig.update_layout(
        **_PLOTLY_LAYOUT_DEFAULTS,
        height=330,
        title=dict(text="7-Day Continuous Glucose Monitor", font=dict(size=14, color="#c9d1d9")),
        xaxis=dict(title="Time", gridcolor="#21262d"),
        yaxis=dict(title="Glucose (mg/dL)", range=[30, 420], gridcolor="#21262d"),
        showlegend=False,
    )
    return fig


def _make_progression_chart(probs: Dict[str, List[float]], years: List[int]) -> go.Figure:
    """Stacked area chart of 5-year DR stage progression probabilities."""
    fig = go.Figure()
    grade_display = [
        ("No DR", DR_GRADE_COLORS[0]),
        ("Mild NPDR", DR_GRADE_COLORS[1]),
        ("Moderate NPDR", DR_GRADE_COLORS[2]),
        ("Severe NPDR", DR_GRADE_COLORS[3]),
        ("Proliferative DR", DR_GRADE_COLORS[4]),
    ]
    for label, color in grade_display:
        if label in probs:
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=probs[label],
                    mode="lines",
                    stackgroup="one",
                    name=label,
                    line=dict(width=0.5, color=color),
                    fillcolor=color.replace(")", ",0.55)").replace("rgb", "rgba")
                    if color.startswith("rgb")
                    else color + "88",
                )
            )

    fig.update_layout(
        **_PLOTLY_LAYOUT_DEFAULTS,
        height=340,
        title=dict(text="5-Year DR Progression Forecast", font=dict(size=14, color="#c9d1d9")),
        xaxis=dict(title="Year", dtick=1, gridcolor="#21262d"),
        yaxis=dict(title="Probability", range=[0, 1.05], gridcolor="#21262d"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.30, xanchor="center", x=0.5,
                    font=dict(size=11)),
    )
    return fig


# ===================================================================
# Main dashboard
# ===================================================================


def main() -> None:  # noqa: C901 — dashboard wiring is inherently complex
    """Entry-point for the Streamlit DR Digital Twin dashboard."""

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    st.markdown(
        """
        <div style="text-align:center; padding: 12px 0 4px 0;">
            <h1 style="margin:0; color:#e6edf3;">🏥 Diabetic Retinopathy Digital Twin</h1>
            <p style="color:#58a6ff; font-size:1.15rem; margin:4px 0 0 0; font-weight:600;">
                AI-Powered Progression Prediction &amp; Personalized Screening
            </p>
            <p style="color:#8b949e; font-size:.92rem; max-width:720px; margin:6px auto 0 auto;">
                Upload a fundus photograph and enter patient data to generate a comprehensive
                risk profile, 5-year progression forecast, and evidence-based screening
                recommendation — all powered by a patient-specific digital twin.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ------------------------------------------------------------------
    # Sidebar — Patient input
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### 📋 Patient Information")
        uploaded_file = st.file_uploader(
            "Fundus Image",
            type=["jpg", "jpeg", "png"],
            help="Upload a retinal fundus photograph for AI grading.",
        )
        st.divider()

        patient_name = st.text_input("Patient Name", value="Patient Demo")
        age = st.slider("Age", 35, 75, 55)
        gender = st.selectbox("Gender", ["M", "F"])
        diabetes_duration = st.slider("Diabetes Duration (years)", 1, 25, 8)
        hba1c = st.slider("HbA1c (%)", 5.0, 14.0, 8.5, step=0.1)
        fasting_glucose = st.slider("Fasting Glucose (mg/dL)", 80, 400, 180)
        systolic_bp = st.slider("Systolic BP (mmHg)", 100, 200, 140)
        diastolic_bp = st.slider("Diastolic BP (mmHg)", 60, 120, 85)
        bmi = st.slider("BMI", 18.0, 45.0, 28.0, step=0.5)
        egfr = st.slider("eGFR (mL/min/1.73m²)", 15, 120, 75)
        med_adherence = st.slider("Medication Adherence (%)", 0, 100, 65)
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

        st.divider()
        run_analysis = st.button("🔬 Run Analysis", use_container_width=True, type="primary")

    # ------------------------------------------------------------------
    # Assemble patient record
    # ------------------------------------------------------------------
    patient = generate_patient(overrides={
        "name": patient_name,
        "age": age,
        "gender": gender,
        "diabetes_duration_years": diabetes_duration,
        "hba1c": hba1c,
        "fasting_glucose_mg_dl": fasting_glucose,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "bmi": bmi,
        "egfr": egfr,
        "medication_adherence_pct": med_adherence,
        "smoking_status": smoking,
    })

    # ------------------------------------------------------------------
    # Run analysis gate
    # ------------------------------------------------------------------
    if not run_analysis:
        st.info(
            "👈 Adjust patient parameters in the sidebar, then press **🔬 Run Analysis** "
            "to generate the digital twin report.",
            icon="ℹ️",
        )
        return

    # ------------------------------------------------------------------
    # Fundus grading (model or demo mode)
    # ------------------------------------------------------------------
    fundus_image: Image.Image | None = None
    gradcam_overlay: np.ndarray | None = None  # type: ignore[type-arg]
    dr_grade: int
    grade_confidence: float

    if uploaded_file is not None:
        fundus_image = Image.open(uploaded_file).convert("RGB")

    model_loaded = False
    if MODEL_AVAILABLE and os.path.isfile(CHECKPOINT_PATH) and fundus_image is not None:
        try:
            import torch, cv2
            device = "cuda" if torch.cuda.is_available() else "cpu"
            grader = load_model(checkpoint_path=CHECKPOINT_PATH, device=device)
            explainer = GradCAMExplainer(grader, device=device)
            img_array = np.array(fundus_image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            result = predict_image(img_bgr, model=grader, device=device)
            dr_grade = result["grade"]
            grade_confidence = result["confidence"]
            heatmap, _, _ = explainer.generate_heatmap(img_bgr)
            gradcam_overlay = explainer.overlay_on_image(img_bgr, heatmap)
            gradcam_overlay = cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB)
            model_loaded = True
        except Exception:
            logger.exception("Model inference failed — falling back to demo mode")
            model_loaded = False

    if not model_loaded:
        dr_grade = _estimate_dr_grade_from_ehr(hba1c, diabetes_duration)
        grade_confidence = round(random.uniform(0.72, 0.96), 2)

    # ------------------------------------------------------------------
    # CGM / wearable data
    # ------------------------------------------------------------------
    cgm = generate_cgm_stream(hba1c=hba1c, duration_days=7)
    wearable_summary = generate_wearable_summary(cgm)

    # ------------------------------------------------------------------
    # Digital-twin computations
    # ------------------------------------------------------------------
    patient["dr_grade"] = dr_grade
    risk_calc = RiskCalculator()
    risk_result = risk_calc.compute_risk_score(patient, wearable_summary)

    progression_model = DRProgressionModel()
    progression = progression_model.simulate_progression(
        current_grade=dr_grade, risk_score=risk_result["total_score"], years=5,
    )

    recommender = ScreeningRecommender()
    recommendation = recommender.recommend(
        risk_score=risk_result["total_score"],
        current_grade=dr_grade,
        progression_data=progression,
    )

    twin = DigitalTwin(patient.get("patient_id", "DEMO-001"))
    twin.ingest_ehr(patient)
    twin.set_fundus_result(dr_grade, grade_confidence)
    twin.risk_assessment = risk_result
    twin.progression_forecast = progression
    twin.screening_recommendation = recommendation
    summary_text = twin.get_summary_text()
    wearable_summary = generate_wearable_summary(cgm)

    # ------------------------------------------------------------------
    # Layout — 2 × 2 grid
    # ------------------------------------------------------------------
    top_left, top_right = st.columns(2, gap="large")
    bottom_left, bottom_right = st.columns(2, gap="large")

    # === Panel A — Fundus Analysis ====================================
    with top_left:
        st.markdown('<div class="panel-card"><div class="panel-title">👁️ Fundus Analysis</div>',
                    unsafe_allow_html=True)

        if fundus_image is not None and model_loaded:
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(fundus_image, caption="Original Fundus", use_container_width=True)
            with img_col2:
                if gradcam_overlay is not None:
                    st.image(gradcam_overlay, caption="GradCAM Overlay",
                             use_container_width=True)
        elif fundus_image is not None:
            st.image(fundus_image, caption="Uploaded Fundus (demo grading)",
                     use_container_width=True)
            st.caption("ℹ️ Model checkpoint not found — grade estimated from EHR data.")
        else:
            st.markdown(
                '<p style="color:#8b949e;">No fundus image uploaded. '
                "Running in <b>demo mode</b> — DR grade estimated from clinical data.</p>",
                unsafe_allow_html=True,
            )

        st.markdown(_grade_badge_html(dr_grade), unsafe_allow_html=True)
        _build_confidence_bar(grade_confidence)
        st.markdown("</div>", unsafe_allow_html=True)

    # === Panel B — Risk Assessment ====================================
    with top_right:
        st.markdown(
            '<div class="panel-card"><div class="panel-title">⚠️ Risk Assessment</div>',
            unsafe_allow_html=True,
        )

        risk_score: float = risk_result.get("total_score", 0.0)
        risk_category: str = risk_result.get("category", "Unknown")
        factor_breakdown: Dict[str, Any] = risk_result.get("factor_breakdown", {})

        st.plotly_chart(_make_gauge_chart(risk_score), use_container_width=True)

        cat_color = "#238636" if risk_score < 25 else (
            "#d29922" if risk_score < 50 else (
                "#da3633" if risk_score < 75 else "#8b0000"
            )
        )
        st.markdown(
            f'<p style="text-align:center;font-size:1.35rem;font-weight:800;'
            f'color:{cat_color};margin:0;">{risk_category}</p>',
            unsafe_allow_html=True,
        )

        if factor_breakdown:
            # Extract scores for radar chart from factor_breakdown
            factor_scores = {v.get("label", k): v.get("score", 0) * 100 for k, v in factor_breakdown.items() if isinstance(v, dict)}
            st.plotly_chart(_make_radar_chart(factor_scores), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # === Panel C — Wearable / CGM data ================================
    with bottom_left:
        st.markdown(
            '<div class="panel-card"><div class="panel-title">📱 Wearable Data — CGM</div>',
            unsafe_allow_html=True,
        )

        timestamps = cgm["timestamp"].tolist() if "timestamp" in cgm.columns else []
        glucose_values = cgm["glucose_mg_dl"].tolist() if "glucose_mg_dl" in cgm.columns else []
        if timestamps and glucose_values:
            st.plotly_chart(_make_cgm_chart(timestamps, glucose_values),
                           use_container_width=True)

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Avg Glucose", f"{wearable_summary.get('avg_glucose', 0):.0f} mg/dL")
        mc2.metric("Time-in-Range", f"{wearable_summary.get('time_in_range_pct', 0):.0f}%")
        mc3.metric("Glucose CV", f"{wearable_summary.get('glucose_variability', 0):.1f}%")

        st.markdown("</div>", unsafe_allow_html=True)

    # === Panel D — Progression Forecast ===============================
    with bottom_right:
        st.markdown(
            '<div class="panel-card"><div class="panel-title">📈 Progression Forecast</div>',
            unsafe_allow_html=True,
        )

        yearly_probs = progression.get("yearly_probabilities", {})
        years_list = sorted(yearly_probs.keys()) if yearly_probs else list(range(6))
        if yearly_probs:
            # Convert to format expected by chart: {"No DR": [prob_y0, ...], "Mild NPDR": [...], ...}
            stage_probs = {}
            for grade in range(5):
                label = DR_GRADE_LABELS[grade]
                stage_probs[label] = [yearly_probs.get(y, {}).get(grade, 0.0) for y in years_list]
            st.plotly_chart(_make_progression_chart(stage_probs, years_list), use_container_width=True)

        # Screening recommendation box
        interval = recommendation.get("screening_interval_months", "—")
        referral = recommendation.get("referral_type", "—")
        urgency = recommendation.get("urgency", "Routine")
        lifestyle = recommendation.get("lifestyle_recommendations", [])

        urgency_colors = {
            "Routine": "#238636",
            "Priority": "#d29922",
            "Urgent": "#da3633",
            "Emergent": "#8b0000",
            "Emergency": "#8b0000",
        }
        urg_color = urgency_colors.get(urgency, "#58a6ff")

        rec_html = f"""
        <div class="rec-box">
            <h4>📋 Screening Recommendation</h4>
            <p class="rec-item">🕐 <b>Next screening:</b> {interval} months</p>
            <p class="rec-item">🏥 <b>Referral:</b> {referral}</p>
            <p class="rec-item">
                <span style="display:inline-block;width:10px;height:10px;
                             border-radius:50%;background:{urg_color};
                             margin-right:4px;"></span>
                <b>Urgency:</b> {urgency}
            </p>
        """
        if lifestyle:
            rec_html += '<p class="rec-item" style="margin-top:8px;"><b>Lifestyle:</b></p>'
            for tip in lifestyle:
                rec_html += f'<p class="rec-item">  • {tip}</p>'
        rec_html += "</div>"
        st.markdown(rec_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Bottom — Clinical summary
    # ------------------------------------------------------------------
    st.divider()
    with st.expander("📄 Full Clinical Summary", expanded=False):
        st.markdown(summary_text)

    st.markdown(
        """
        <div class="disclaimer">
            <b>Data Sources:</b> Synthetic EHR, simulated CGM, AI fundus grading.<br>
            <b>Disclaimer:</b> This is a proof-of-concept demonstration system.
            <b>Not for clinical decision-making.</b> All patient data shown is synthetic.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
