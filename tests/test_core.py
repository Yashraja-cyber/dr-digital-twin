"""Quick verification script for all DR Digital Twin modules."""
import sys 
sys .path .insert (0 ,'.')

print ("="*60 )
print ("DR DIGITAL TWIN — MODULE VERIFICATION")
print ("="*60 )


from src .config import CLASS_NAMES ,DR_TRANSITION_RATES ,RISK_WEIGHTS 
print ("[OK] src.config loaded")


from src .data .synthetic_ehr import generate_patient ,generate_cgm_stream ,generate_wearable_summary 
patient =generate_patient ("PT-001",dr_grade =2 )
print (f"[OK] generate_patient — age={patient['age']}, HbA1c={patient['hba1c']}, DR grade={patient['dr_grade']}")


cgm =generate_cgm_stream (patient ['hba1c'],duration_days =7 )
print (f"[OK] generate_cgm_stream — {len(cgm)} readings, cols={list(cgm.columns)}")


ws =generate_wearable_summary (cgm )
print (f"[OK] generate_wearable_summary — TIR={ws['time_in_range_pct']}%, avg={ws['avg_glucose']}")


from src .digital_twin .risk_calculator import RiskCalculator 
rc =RiskCalculator ()
risk =rc .compute_risk_score (patient ,ws )
print (f"[OK] RiskCalculator — score={risk['total_score']}/100, category={risk['category']}")


from src .digital_twin .progression_model import DRProgressionModel 
pm =DRProgressionModel ()
prog =pm .simulate_progression (current_grade =2 ,risk_score =risk ['total_score'],years =5 )
print (f"[OK] DRProgressionModel — P(progression)={prog['probability_of_progression']:.1%}, worst_case={prog['worst_case_grade']}")


from src .digital_twin .screening_recommender import ScreeningRecommender 
sr =ScreeningRecommender ()
rec =sr .recommend (risk ['total_score'],2 ,prog )
print (f"[OK] ScreeningRecommender — interval={rec['screening_interval_months']}mo, urgency={rec['urgency']}")


from src .digital_twin .twin_engine import DigitalTwin 
twin =DigitalTwin ("PT-001")
twin .ingest_ehr (patient )
twin .ingest_wearable (cgm )
analysis =twin .run_full_analysis ()
print (f"[OK] DigitalTwin — full analysis complete")
print (f"     Risk: {analysis['risk_assessment']['total_score']}/100 ({analysis['risk_assessment']['category']})")
prog_result =analysis ['progression_forecast']
print (f"     Progression: {prog_result['probability_of_progression']:.1%} over 5 years")
screen =analysis ['screening_recommendation']
print (f"     Screening: every {screen['screening_interval_months']}mo, {screen['urgency']}")


summary =twin .get_summary_text ()
print (f"[OK] Clinical summary generated ({len(summary)} chars)")

print ()
print ("="*60 )
print ("ALL CORE MODULES VERIFIED SUCCESSFULLY")
print ("="*60 )
