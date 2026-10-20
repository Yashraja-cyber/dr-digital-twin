
"""Synthetic EHR and wearable/CGM data generator for Diabetic Retinopathy research.

Generates clinically realistic patient records correlated with DR severity
grades (0–4), continuous glucose monitoring streams with circadian physiology,
and wearable-derived summary metrics.  All distributions are calibrated to
Indian epidemiological baselines (ICMR-INDIAB, CURES, Raman et al. 2009).

Typical usage::

    from src.data.synthetic_ehr import (
        generate_patient,
        generate_cgm_stream,
        generate_wearable_summary,
        generate_patient_cohort,
    )

    cohort = generate_patient_cohort(n_patients=500, seed=42)
    pat    = generate_patient("P-0001", dr_grade=3)
    cgm    = generate_cgm_stream(pat["hba1c"], duration_days=14)
    ws     = generate_wearable_summary(cgm)
"""

from __future__ import annotations 

import logging 
from typing import Any 

import numpy as np 
import pandas as pd 
from numpy .random import Generator 
from scipy import stats 

logger =logging .getLogger (__name__ )


DR_GRADE_LABELS :dict [int ,str ]={
0 :"No DR",
1 :"Mild NPDR",
2 :"Moderate NPDR",
3 :"Severe NPDR",
4 :"Proliferative DR",
}


_GRADE_PROBS :list [float ]=[0.50 ,0.18 ,0.15 ,0.10 ,0.07 ]


_ORAL_MEDS :list [str ]=[
"Metformin",
"Glimepiride",
"Gliclazide",
"Sitagliptin",
"Vildagliptin",
"Empagliflozin",
"Dapagliflozin",
"Pioglitazone",
"Teneligliptin",
"Voglibose",
]

_INSULIN_TYPES :list [str ]=[
"Insulin Glargine",
"Insulin Aspart",
"Insulin Lispro",
"NPH Insulin",
"Premixed Insulin 30/70",
]

_ANTIHYPERTENSIVES :list [str ]=[
"Telmisartan",
"Amlodipine",
"Losartan",
"Ramipril",
"Cilnidipine",
"Chlorthalidone",
]

_LIPID_LOWERING :list [str ]=[
"Atorvastatin",
"Rosuvastatin",
"Fenofibrate",
]

_COMORBIDITY_POOL :list [str ]=[
"Hypertension",
"Dyslipidemia",
"Coronary Artery Disease",
"Chronic Kidney Disease",
"Peripheral Neuropathy",
"Diabetic Foot",
"Diabetic Nephropathy",
"Hypothyroidism",
"Non-Alcoholic Fatty Liver Disease",
"Obstructive Sleep Apnea",
"Obesity",
"Cerebrovascular Disease",
]


def _clamp (value :float ,lo :float ,hi :float )->float :
    """Clamp *value* between *lo* and *hi* inclusive."""
    return max (lo ,min (hi ,value ))


def _pick_medications (rng :Generator ,grade :int )->list [str ]:
    """Return a clinically plausible medication list correlated to DR grade.

    Higher grades → more medications and higher insulin probability.
    """
    meds :list [str ]=[]


    if rng .random ()<0.92 :
        meds .append ("Metformin")


    n_orals =int (rng .integers (0 ,min (grade +2 ,4 )))
    candidates =[m for m in _ORAL_MEDS if m not in meds ]
    rng .shuffle (candidates )
    meds .extend (candidates [:n_orals ])


    insulin_prob =[0.05 ,0.15 ,0.35 ,0.60 ,0.80 ][grade ]
    if rng .random ()<insulin_prob :
        meds .append (rng .choice (_INSULIN_TYPES ))


    if rng .random ()<0.3 +grade *0.12 :
        meds .append (rng .choice (_ANTIHYPERTENSIVES ))


    if rng .random ()<0.25 +grade *0.10 :
        meds .append (rng .choice (_LIPID_LOWERING ))

    return list (dict .fromkeys (meds ))


def _pick_comorbidities (rng :Generator ,grade :int )->list [str ]:
    """Return comorbidities; number & severity increase with grade."""
    n =int (rng .integers (0 ,min (grade +2 ,6 )))

    weights =np .ones (len (_COMORBIDITY_POOL ))
    weights [0 ]=3.0 
    weights [1 ]=2.5 
    if grade >=3 :
        weights [3 ]=2.0 
        weights [4 ]=2.0 
        weights [6 ]=2.0 
    probs =weights /weights .sum ()
    chosen =rng .choice (
    _COMORBIDITY_POOL ,size =min (n ,len (_COMORBIDITY_POOL )),replace =False ,p =probs 
    )
    return list (chosen )


_PARAM_TABLE :dict [str ,list [tuple [float ,float ]]]={
"age":[
(45.0 ,8.0 ),
(50.0 ,7.0 ),
(54.0 ,7.0 ),
(58.0 ,6.5 ),
(62.0 ,6.0 ),
],
"diabetes_duration_years":[
(4.0 ,2.0 ),
(7.0 ,2.5 ),
(10.0 ,3.0 ),
(14.0 ,3.5 ),
(18.0 ,4.0 ),
],
"hba1c":[
(7.0 ,0.5 ),
(7.8 ,0.6 ),
(8.5 ,0.8 ),
(9.5 ,1.0 ),
(10.5 ,1.2 ),
],
"fasting_glucose_mg_dl":[
(120.0 ,15.0 ),
(145.0 ,20.0 ),
(170.0 ,30.0 ),
(210.0 ,40.0 ),
(260.0 ,50.0 ),
],
"systolic_bp":[
(126.0 ,8.0 ),
(132.0 ,10.0 ),
(140.0 ,12.0 ),
(152.0 ,14.0 ),
(164.0 ,16.0 ),
],
"diastolic_bp":[
(78.0 ,6.0 ),
(82.0 ,7.0 ),
(86.0 ,8.0 ),
(90.0 ,9.0 ),
(96.0 ,10.0 ),
],
"bmi":[
(25.5 ,3.0 ),
(27.0 ,3.5 ),
(28.5 ,3.5 ),
(29.5 ,3.5 ),
(30.5 ,4.0 ),
],
"total_cholesterol_mg_dl":[
(190.0 ,25.0 ),
(200.0 ,28.0 ),
(215.0 ,30.0 ),
(230.0 ,32.0 ),
(250.0 ,35.0 ),
],
"ldl_mg_dl":[
(105.0 ,20.0 ),
(115.0 ,22.0 ),
(125.0 ,25.0 ),
(140.0 ,28.0 ),
(155.0 ,30.0 ),
],
"hdl_mg_dl":[
(50.0 ,8.0 ),
(47.0 ,8.0 ),
(44.0 ,7.0 ),
(41.0 ,7.0 ),
(38.0 ,6.0 ),
],
"triglycerides_mg_dl":[
(140.0 ,30.0 ),
(160.0 ,35.0 ),
(180.0 ,40.0 ),
(210.0 ,50.0 ),
(240.0 ,55.0 ),
],
"egfr_ml_min":[
(100.0 ,12.0 ),
(90.0 ,14.0 ),
(78.0 ,15.0 ),
(62.0 ,16.0 ),
(48.0 ,14.0 ),
],
"serum_creatinine":[
(0.85 ,0.12 ),
(0.95 ,0.15 ),
(1.10 ,0.20 ),
(1.35 ,0.30 ),
(1.70 ,0.40 ),
],
"medication_adherence_pct":[
(85.0 ,8.0 ),
(78.0 ,10.0 ),
(70.0 ,12.0 ),
(62.0 ,14.0 ),
(55.0 ,15.0 ),
],
}


_CLAMP_RANGES :dict [str ,tuple [float ,float ]]={
"age":(35 ,75 ),
"diabetes_duration_years":(2 ,25 ),
"hba1c":(6.5 ,14.0 ),
"fasting_glucose_mg_dl":(100 ,350 ),
"systolic_bp":(120 ,180 ),
"diastolic_bp":(70 ,110 ),
"bmi":(22.0 ,38.0 ),
"total_cholesterol_mg_dl":(150 ,300 ),
"ldl_mg_dl":(60 ,220 ),
"hdl_mg_dl":(25 ,75 ),
"triglycerides_mg_dl":(80 ,400 ),
"egfr_ml_min":(30 ,120 ),
"serum_creatinine":(0.5 ,3.5 ),
"medication_adherence_pct":(40 ,95 ),
}


def generate_patient (
patient_id :str ="P-0001",
dr_grade :int |None =None ,
overrides :dict [str ,Any ]|None =None ,
*,
rng :Generator |None =None ,
)->dict [str ,Any ]:
    """Generate a single synthetic patient record.

    Parameters
    ----------
    patient_id:
        Unique identifier for the patient (e.g. ``"P-0001"``).
    dr_grade:
        Diabetic Retinopathy severity grade (0–4).  If *None*, a grade is
        sampled from the population distribution.
    overrides:
        Optional dictionary of field values to override (e.g. from UI sliders).
    rng:
        Optional :class:`numpy.random.Generator` for reproducibility.

    Returns
    -------
    dict
        A dictionary containing all EHR fields including demographics,
        lab values, medications, comorbidities, and DR grade information.
    """
    if rng is None :
        rng =np .random .default_rng ()

    if dr_grade is None :
        dr_grade =int (rng .choice (5 ,p =_GRADE_PROBS ))
    else :
        if dr_grade not in DR_GRADE_LABELS :
            raise ValueError (
            f"dr_grade must be 0–4, got {dr_grade}"
            )


    values :dict [str ,float ]={}
    for param ,grade_dists in _PARAM_TABLE .items ():
        mean ,sd =grade_dists [dr_grade ]
        raw =float (rng .normal (mean ,sd ))
        lo ,hi =_CLAMP_RANGES [param ]
        values [param ]=round (_clamp (raw ,lo ,hi ),2 )


    gender :str =rng .choice (["M","F"],p =[0.58 ,0.42 ])

    diabetes_type :str =rng .choice (
    ["Type 2","Type 1"],p =[0.95 ,0.05 ]
    )
    smoking_status :str =rng .choice (
    ["never","former","current"],
    p =[0.60 ,0.25 ,0.15 ],
    )
    family_history :bool =bool (rng .random ()<(0.45 +dr_grade *0.05 ))

    record :dict [str ,Any ]={
    "patient_id":patient_id ,
    "age":int (values ["age"]),
    "gender":gender ,
    "diabetes_duration_years":round (values ["diabetes_duration_years"],1 ),
    "diabetes_type":diabetes_type ,
    "hba1c":round (values ["hba1c"],1 ),
    "fasting_glucose_mg_dl":round (values ["fasting_glucose_mg_dl"],1 ),
    "systolic_bp":int (values ["systolic_bp"]),
    "diastolic_bp":int (values ["diastolic_bp"]),
    "bmi":round (values ["bmi"],1 ),
    "total_cholesterol_mg_dl":round (values ["total_cholesterol_mg_dl"],1 ),
    "ldl_mg_dl":round (values ["ldl_mg_dl"],1 ),
    "hdl_mg_dl":round (values ["hdl_mg_dl"],1 ),
    "triglycerides_mg_dl":round (values ["triglycerides_mg_dl"],1 ),
    "egfr_ml_min":round (values ["egfr_ml_min"],1 ),
    "serum_creatinine":round (values ["serum_creatinine"],2 ),
    "smoking_status":smoking_status ,
    "family_history_diabetes":family_history ,
    "medication_adherence_pct":round (values ["medication_adherence_pct"],1 ),
    "medications":_pick_medications (rng ,dr_grade ),
    "comorbidities":_pick_comorbidities (rng ,dr_grade ),
    "dr_grade":dr_grade ,
    "dr_grade_label":DR_GRADE_LABELS [dr_grade ],
    }
    if overrides :
        for k ,v in overrides .items ():
            if v is not None :
                record [k ]=v 
        if "dr_grade"in overrides and overrides ["dr_grade"]in DR_GRADE_LABELS :
            record ["dr_grade_label"]=DR_GRADE_LABELS [overrides ["dr_grade"]]
    return record 


def generate_cgm_stream (
hba1c :float ,
duration_days :int =30 ,
readings_per_day :int =288 ,
*,
rng :Generator |None =None ,
)->pd .DataFrame :
    """Generate a continuous glucose monitoring stream with wearable vitals.

    The glucose signal is built from:

    1. **Baseline** derived from HbA1c via the ADAG formula
       (estimated_avg_glucose = 28.7 × HbA1c − 46.7).
    2. **Circadian component** – dawn phenomenon (4–8 AM), post-prandial
       spikes after breakfast (8 AM), lunch (1 PM), dinner (7 PM).
    3. **Gaussian noise** (SD 15–25 mg/dL, higher for higher HbA1c).
    4. **Random-walk drift** for realistic autocorrelation.
    5. **Hypoglycaemic events** – occasional dips below 70 for poorly
       controlled patients on intensive therapy.

    Parameters
    ----------
    hba1c:
        Patient's HbA1c value used to anchor the glucose baseline.
    duration_days:
        Number of days of CGM data to generate.
    readings_per_day:
        Readings per day (288 = every 5 min).
    rng:
        Optional :class:`numpy.random.Generator` for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp``, ``glucose_mg_dl``, ``heart_rate_bpm``,
        ``steps_hourly``.
    """
    if rng is None :
        rng =np .random .default_rng ()

    total_readings =duration_days *readings_per_day 
    interval_minutes =1440.0 /readings_per_day 


    start =pd .Timestamp ("2024-01-01")
    timestamps =pd .date_range (
    start =start ,
    periods =total_readings ,
    freq =pd .Timedelta (minutes =interval_minutes ),
    )


    baseline =28.7 *hba1c -46.7 


    hour_of_day =(
    timestamps .hour +timestamps .minute /60.0 +timestamps .second /3600.0 
    )


    circadian =np .zeros (total_readings )


    dawn_mask =(hour_of_day >=4 )&(hour_of_day <8 )
    dawn_center =6.0 
    circadian [dawn_mask ]=20.0 *np .exp (
    -0.5 *((hour_of_day [dawn_mask ]-dawn_center )/1.0 )**2 
    )


    breakfast_mask =(hour_of_day >=8 )&(hour_of_day <11 )
    circadian [breakfast_mask ]=45.0 *np .exp (
    -0.5 *((hour_of_day [breakfast_mask ]-9.0 )/0.8 )**2 
    )


    lunch_mask =(hour_of_day >=13 )&(hour_of_day <16 )
    circadian [lunch_mask ]=40.0 *np .exp (
    -0.5 *((hour_of_day [lunch_mask ]-14.0 )/0.8 )**2 
    )


    dinner_mask =(hour_of_day >=19 )&(hour_of_day <22 )
    circadian [dinner_mask ]=42.0 *np .exp (
    -0.5 *((hour_of_day [dinner_mask ]-20.0 )/0.9 )**2 
    )


    severity_factor =0.6 +0.15 *(hba1c -6.5 )
    circadian *=severity_factor 


    noise_sd =_clamp (15.0 +(hba1c -7.0 )*3.0 ,15.0 ,30.0 )
    innovations =rng .normal (0 ,noise_sd *0.15 ,total_readings )
    drift =np .cumsum (innovations )

    alpha =0.005 
    for i in range (1 ,total_readings ):
        drift [i ]=drift [i -1 ]*(1 -alpha )+innovations [i ]


    noise =rng .normal (0 ,noise_sd ,total_readings )


    glucose =baseline +circadian +drift +noise 


    if hba1c >=8.5 :
        hypo_prob =0.0015 *(hba1c -8.0 )
        hypo_mask =rng .random (total_readings )<hypo_prob 
        glucose [hypo_mask ]=rng .uniform (45 ,68 ,int (hypo_mask .sum ()))


    glucose =np .clip (glucose ,40.0 ,500.0 )
    glucose =np .round (glucose ,1 )


    resting_hr =rng .uniform (62 ,88 )

    hr_circadian =np .where (
    (hour_of_day >=23 )|(hour_of_day <6 ),
    -8.0 ,
    np .where (
    (hour_of_day >=10 )&(hour_of_day <17 ),
    5.0 ,
    0.0 ,
    ),
    )
    hr_noise =rng .normal (0 ,4.0 ,total_readings )
    heart_rate =np .clip (resting_hr +hr_circadian +hr_noise ,50 ,130 ).astype (int )


    steps_base =np .where (
    (hour_of_day >=23 )|(hour_of_day <6 ),
    0.0 ,
    np .where (
    (hour_of_day >=7 )&(hour_of_day <9 ),
    rng .poisson (150 ,total_readings if isinstance (hour_of_day ,np .ndarray )else 1 ),
    np .where (
    (hour_of_day >=17 )&(hour_of_day <19 ),
    rng .poisson (250 ,total_readings ),
    rng .poisson (80 ,total_readings ),
    ),
    ),
    )

    steps_base =np .where (
    (hour_of_day >=23 )|(hour_of_day <6 ),
    0 ,
    steps_base ,
    )
    steps_hourly =np .clip (steps_base ,0 ,1500 ).astype (int )

    df =pd .DataFrame (
    {
    "timestamp":timestamps ,
    "glucose_mg_dl":glucose ,
    "heart_rate_bpm":heart_rate ,
    "steps_hourly":steps_hourly ,
    }
    )
    logger .debug (
    "Generated CGM stream: %d days, %d total readings, baseline=%.1f",
    duration_days ,
    total_readings ,
    baseline ,
    )
    return df 


def generate_wearable_summary (cgm_df :pd .DataFrame )->dict [str ,float ]:
    """Derive clinically meaningful wearable summary metrics from CGM data.

    Implements the International Consensus on use of CGM metrics
    (Battelino et al. Diabetes Care, 2019).

    Parameters
    ----------
    cgm_df:
        DataFrame returned by :func:`generate_cgm_stream` with a
        ``glucose_mg_dl`` column.

    Returns
    -------
    dict
        Keys: ``time_in_range_pct``, ``time_above_range_pct``,
        ``time_below_range_pct``, ``avg_glucose``, ``glucose_variability``,
        ``estimated_a1c``, ``avg_heart_rate``, ``avg_daily_steps``,
        ``glucose_management_indicator``.

    Raises
    ------
    ValueError
        If the input DataFrame is empty or missing required columns.
    """
    required_cols ={"glucose_mg_dl","heart_rate_bpm","steps_hourly","timestamp"}
    missing =required_cols -set (cgm_df .columns )
    if missing :
        raise ValueError (f"CGM DataFrame is missing columns: {missing}")
    if cgm_df .empty :
        raise ValueError ("CGM DataFrame is empty")

    glucose =cgm_df ["glucose_mg_dl"].values 
    n =len (glucose )

    in_range =np .sum ((glucose >=70 )&(glucose <=180 ))
    above =np .sum (glucose >180 )
    below =np .sum (glucose <70 )

    avg_glucose =float (np .mean (glucose ))
    sd_glucose =float (np .std (glucose ))
    cv_pct =round ((sd_glucose /avg_glucose )*100 ,1 )if avg_glucose >0 else 0.0 


    estimated_a1c =round ((avg_glucose +46.7 )/28.7 ,1 )


    gmi =round (3.31 +0.02392 *avg_glucose ,1 )


    cgm_with_day =cgm_df .copy ()
    cgm_with_day ["date"]=pd .to_datetime (cgm_df ["timestamp"]).dt .date 
    daily_steps =cgm_with_day .groupby ("date")["steps_hourly"].sum ()
    avg_daily_steps =round (float (daily_steps .mean ()),0 )

    summary :dict [str ,float ]={
    "time_in_range_pct":round (100.0 *in_range /n ,1 ),
    "time_above_range_pct":round (100.0 *above /n ,1 ),
    "time_below_range_pct":round (100.0 *below /n ,1 ),
    "avg_glucose":round (avg_glucose ,1 ),
    "glucose_variability":cv_pct ,
    "estimated_a1c":estimated_a1c ,
    "avg_heart_rate":round (float (cgm_df ["heart_rate_bpm"].mean ()),1 ),
    "avg_daily_steps":avg_daily_steps ,
    "glucose_management_indicator":gmi ,
    }
    logger .debug ("Wearable summary: TIR=%.1f%%, GMI=%.1f",summary ["time_in_range_pct"],gmi )
    return summary 


def generate_patient_cohort (
n_patients :int =100 ,
seed :int =42 ,
)->pd .DataFrame :
    """Generate a cohort of synthetic patients with realistic DR grade distribution.

    Parameters
    ----------
    n_patients:
        Number of patients to generate.
    seed:
        Random seed for full reproducibility.

    Returns
    -------
    pd.DataFrame
        Each row is a patient record with all EHR fields.  List-type
        columns (``medications``, ``comorbidities``) are stored as Python
        lists within the DataFrame cells.
    """
    rng =np .random .default_rng (seed )
    records :list [dict [str ,Any ]]=[]

    for i in range (n_patients ):
        pid =f"P-{i:04d}"
        record =generate_patient (pid ,rng =rng )
        records .append (record )

    df =pd .DataFrame (records )
    grade_counts =df ["dr_grade"].value_counts ().sort_index ()
    logger .info (
    "Generated cohort of %d patients. Grade distribution:\n%s",
    n_patients ,
    grade_counts .to_string (),
    )
    return df 


def _main ()->None :
    """Quick smoke-test when run as ``python -m src.data.synthetic_ehr``."""
    logging .basicConfig (level =logging .INFO ,format ="%(levelname)s: %(message)s")

    logger .info ("Generating patient cohort (n=20)...")
    cohort =generate_patient_cohort (n_patients =20 ,seed =7 )
    print (cohort [["patient_id","age","hba1c","dr_grade","dr_grade_label"]].to_string (index =False ))

    logger .info ("\nSample patient record:")
    pat =generate_patient ("P-TEST",dr_grade =3 ,rng =np .random .default_rng (0 ))
    for k ,v in pat .items ():
        print (f"  {k}: {v}")

    logger .info ("\nGenerating 7-day CGM stream...")
    cgm =generate_cgm_stream (pat ["hba1c"],duration_days =7 ,rng =np .random .default_rng (1 ))
    print (f"  CGM shape: {cgm.shape}")
    print (f"  Glucose range: {cgm['glucose_mg_dl'].min():.1f} – {cgm['glucose_mg_dl'].max():.1f} mg/dL")

    summary =generate_wearable_summary (cgm )
    logger .info ("\nWearable summary:")
    for k ,v in summary .items ():
        print (f"  {k}: {v}")


if __name__ =="__main__":
    _main ()
