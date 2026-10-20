"""Evidence-based screening and referral recommendation engine.

Screening intervals and referral pathways follow international guidelines
(AAO Preferred Practice Pattern, ICO, NICE NG28) adapted with patient-level
risk stratification produced by
:class:`~digital_twin.risk_calculator.RiskCalculator` and progression
forecasts from :class:`~digital_twin.progression_model.DRProgressionModel`.
"""

from __future__ import annotations 

import logging 
from typing import Any 

logger =logging .getLogger (__name__ )


_TIERS :list [dict [str ,Any ]]=[
{
"risk_range":(0 ,20 ),
"screening_interval_months":24 ,
"urgency":"Routine",
"referral_type":"Primary Care",
},
{
"risk_range":(21 ,40 ),
"screening_interval_months":12 ,
"urgency":"Routine",
"referral_type":"Ophthalmologist",
},
{
"risk_range":(41 ,60 ),
"screening_interval_months":6 ,
"urgency":"Priority",
"referral_type":"Ophthalmologist",
},
{
"risk_range":(61 ,80 ),
"screening_interval_months":3 ,
"urgency":"Urgent",
"referral_type":"Retina Specialist",
},
{
"risk_range":(81 ,100 ),
"screening_interval_months":1 ,
"urgency":"Emergency",
"referral_type":"Retina Specialist",
},
]


_URGENCY_LADDER :list [str ]=[
"Routine",
"Priority",
"Urgent",
"Emergency",
]


_BASE_TESTS :list [str ]=[
"Dilated fundus examination",
"Visual acuity assessment",
]

_MODERATE_TESTS :list [str ]=[
"Optical coherence tomography (OCT)",
"HbA1c and lipid panel",
]

_HIGH_TESTS :list [str ]=[
"Fluorescein angiography (FA)",
"OCT angiography (OCTA)",
"Renal function panel (eGFR, uACR)",
]

_EMERGENCY_TESTS :list [str ]=[
"Urgent B-scan ultrasonography (if media opacities)",
"Wide-field retinal imaging",
]


class ScreeningRecommender :
    """Generate personalised screening recommendations from risk profiles.

    The recommender maps a composite risk score onto a recommendation tier
    and then applies clinical override rules based on the current DR grade
    and the short-term probability of progression.
    """


    def recommend (
    self ,
    risk_score :float ,
    current_grade :int ,
    progression_data :dict [str ,Any ],
    )->dict [str ,Any ]:
        """Produce a complete screening recommendation payload.

        Parameters
        ----------
        risk_score:
            Composite risk score (0-100) from
            :class:`~digital_twin.risk_calculator.RiskCalculator`.
        current_grade:
            Current DR grade (0-4).
        progression_data:
            Output dictionary from
            :meth:`~digital_twin.progression_model.DRProgressionModel.simulate_progression`.

        Returns
        -------
        dict
            Keys: ``screening_interval_months``, ``urgency``,
            ``referral_type``, ``recommended_tests``,
            ``lifestyle_recommendations``, ``medication_suggestions``,
            ``follow_up_notes``.
        """
        tier =self ._select_tier (risk_score )

        screening_interval_months :int =tier ["screening_interval_months"]
        urgency :str =tier ["urgency"]
        referral_type :str =tier ["referral_type"]


        if current_grade >=3 :
            referral_type ="Retina Specialist"
            if screening_interval_months >3 :
                screening_interval_months =3 
            if _URGENCY_LADDER .index (urgency )<_URGENCY_LADDER .index ("Urgent"):
                urgency ="Urgent"


        prog_2yr =self ._two_year_progression_probability (
        progression_data ,current_grade 
        )
        if prog_2yr >0.60 :
            urgency =self ._upgrade_urgency (urgency )

            screening_interval_months =max (1 ,screening_interval_months //2 )


        recommended_tests =self ._select_tests (risk_score ,current_grade )
        lifestyle_recs =self ._lifestyle_recommendations (risk_score ,current_grade )
        medication_suggestions =self ._medication_suggestions (
        risk_score ,current_grade 
        )
        follow_up_notes =self ._build_follow_up_notes (
        risk_score ,
        current_grade ,
        urgency ,
        prog_2yr ,
        screening_interval_months ,
        )

        logger .info (
        "Screening recommendation: interval=%d mo, urgency=%s, "
        "referral=%s (grade=%d, risk=%.1f, 2yr-prog=%.2f)",
        screening_interval_months ,
        urgency ,
        referral_type ,
        current_grade ,
        risk_score ,
        prog_2yr ,
        )

        return {
        "screening_interval_months":screening_interval_months ,
        "urgency":urgency ,
        "referral_type":referral_type ,
        "recommended_tests":recommended_tests ,
        "lifestyle_recommendations":lifestyle_recs ,
        "medication_suggestions":medication_suggestions ,
        "follow_up_notes":follow_up_notes ,
        }


    @staticmethod 
    def _select_tier (risk_score :float )->dict [str ,Any ]:
        """Return the recommendation tier matching *risk_score*."""
        clamped =max (0.0 ,min (100.0 ,risk_score ))
        for tier in _TIERS :
            low ,high =tier ["risk_range"]
            if low <=clamped <=high :
                return tier 

        return _TIERS [-1 ]


    @staticmethod 
    def _upgrade_urgency (current :str )->str :
        """Move urgency one step up the ladder (capped at Emergency)."""
        idx =_URGENCY_LADDER .index (current )
        return _URGENCY_LADDER [min (idx +1 ,len (_URGENCY_LADDER )-1 )]

    @staticmethod 
    def _two_year_progression_probability (
    progression_data :dict [str ,Any ],
    current_grade :int ,
    )->float :
        """Extract the cumulative probability of progression within 2 years.

        Uses the year-2 row in ``yearly_probabilities`` from the
        simulation output.  If year 2 is not available, falls back to
        the overall ``probability_of_progression``.
        """
        yearly =progression_data .get ("yearly_probabilities",{})
        if 2 in yearly :
            year2_probs =yearly [2 ]

            return sum (
            prob 
            for grade ,prob in year2_probs .items ()
            if int (grade )>current_grade 
            )

        return progression_data .get ("probability_of_progression",0.0 )


    @staticmethod 
    def _select_tests (risk_score :float ,current_grade :int )->list [str ]:
        """Build a list of recommended diagnostic tests."""
        tests =list (_BASE_TESTS )
        if risk_score >20 or current_grade >=1 :
            tests .extend (_MODERATE_TESTS )
        if risk_score >60 or current_grade >=3 :
            tests .extend (_HIGH_TESTS )
        if risk_score >80 or current_grade >=4 :
            tests .extend (_EMERGENCY_TESTS )
        return tests 


    @staticmethod 
    def _lifestyle_recommendations (
    risk_score :float ,
    current_grade :int ,
    )->list [str ]:
        """Return lifestyle modification recommendations."""
        recs :list [str ]=[
        "Monitor blood glucose regularly (aim for HbA1c < 7.0 %)",
        "Maintain blood pressure below 130/80 mmHg",
        ]
        if risk_score >30 :
            recs .append (
            "Adopt a Mediterranean or DASH diet rich in omega-3 fatty acids"
            )
            recs .append ("Engage in ≥ 150 min/week of moderate aerobic exercise")
        if risk_score >50 :
            recs .append ("Smoking cessation counselling (if applicable)")
            recs .append ("Limit alcohol intake to ≤ 1 standard drink/day")
        if current_grade >=2 :
            recs .append (
            "Avoid heavy lifting or Valsalva manoeuvres that raise "
            "intraocular pressure"
            )
        if risk_score >70 :
            recs .append (
            "Consider continuous glucose monitoring (CGM) for tighter "
            "glycaemic control"
            )
        return recs 

    @staticmethod 
    def _medication_suggestions (
    risk_score :float ,
    current_grade :int ,
    )->list [str ]:
        """Return medication-related suggestions.

        .. note::
           These are informational suggestions for the clinician and
           are **not** prescriptions.
        """
        meds :list [str ]=[]
        if risk_score >40 :
            meds .append (
            "Review and optimise anti-hyperglycaemic regimen "
            "(consider SGLT2 inhibitors or GLP-1 RAs for cardio-renal benefit)"
            )
        if risk_score >30 :
            meds .append (
            "Consider ACE inhibitor / ARB for nephroprotection and "
            "possible DR risk reduction"
            )
        if risk_score >50 :
            meds .append ("Statin therapy for dyslipidaemia if LDL > 100 mg/dL")
        if current_grade >=3 :
            meds .append (
            "Discuss anti-VEGF intravitreal therapy "
            "(ranibizumab / aflibercept / faricimab) with retina specialist"
            )
        if current_grade >=4 :
            meds .append (
            "Consider panretinal photocoagulation (PRP) or vitrectomy referral"
            )
        if not meds :
            meds .append ("Continue current medication regimen; no changes suggested")
        return meds 


    @staticmethod 
    def _build_follow_up_notes (
    risk_score :float ,
    current_grade :int ,
    urgency :str ,
    prog_2yr :float ,
    interval_months :int ,
    )->str :
        """Compose a concise clinical follow-up note."""
        lines :list [str ]=[]
        lines .append (
        f"Risk score: {risk_score:.1f}/100. "
        f"Current DR grade: {current_grade}/4."
        )
        lines .append (
        f"2-year progression probability: {prog_2yr * 100:.1f} %."
        )
        lines .append (
        f"Recommended screening interval: every {interval_months} month(s) "
        f"(urgency: {urgency})."
        )
        if current_grade >=3 :
            lines .append (
            "⚠ Severe or proliferative DR detected — mandatory retina "
            "specialist referral."
            )
        if prog_2yr >0.60 :
            lines .append (
            "⚠ Elevated 2-year progression risk — urgency upgraded and "
            "screening interval halved."
            )
        if risk_score >80 :
            lines .append (
            "🚨 Critical risk — consider immediate referral and "
            "multidisciplinary review."
            )
        return " ".join (lines )
