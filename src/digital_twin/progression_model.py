"""Markov chain Monte Carlo model for diabetic retinopathy grade progression.

Base annual transition probabilities are derived from longitudinal cohort
studies (WESDR, UKPDS, CURES).  The model uses five DR states:

======  ============================
State   Clinical Grade
======  ============================
0       No DR
1       Mild NPDR
2       Moderate NPDR
3       Severe NPDR
4       Proliferative DR (absorbing)
======  ============================

Key modelling decisions:
- No backward transitions: DR does not spontaneously reverse.
- State 4 is absorbing: once proliferative, the patient stays there.
- Risk-adjusted transitions: a patient's composite risk score modulates
  the base transition matrix via a multiplicative factor.
"""

from __future__ import annotations 

import logging 

import numpy as np 

logger =logging .getLogger (__name__ )


_N_STATES :int =5 


class DRProgressionModel :
    """Markov chain model for simulating DR grade progression.

    The base annual transition matrix encodes the probability of moving
    from one DR state to another in a single year.  A patient-specific
    risk score scales these probabilities up or down before simulation.
    """

    def __init__ (self )->None :
        """Initialise the model with the base (population-level) transition matrix."""

        self ._base_forward :dict [int ,float ]={
        0 :0.05 ,
        1 :0.08 ,
        2 :0.12 ,
        3 :0.15 ,
        }
        self ._base_matrix :np .ndarray =self ._build_base_matrix ()
        logger .debug ("DRProgressionModel initialised with base matrix:\n%s",self ._base_matrix )


    def _build_base_matrix (self )->np .ndarray :
        """Construct the 5×5 base transition matrix.

        Returns
        -------
        np.ndarray
            Row-stochastic matrix where ``M[i][j]`` is the probability
            of transitioning from state *i* to state *j* in one year.
        """
        matrix =np .zeros ((_N_STATES ,_N_STATES ),dtype =np .float64 )
        for state in range (_N_STATES ):
            if state ==_N_STATES -1 :

                matrix [state ,state ]=1.0 
            else :
                forward_prob =self ._base_forward [state ]
                matrix [state ,state +1 ]=forward_prob 
                matrix [state ,state ]=1.0 -forward_prob 
        return matrix 

    def _build_risk_adjusted_matrix (self ,risk_score :float )->np .ndarray :
        """Scale the base transition matrix by a patient-specific risk score.

        Parameters
        ----------
        risk_score:
            Composite risk score on the 0-100 scale produced by
            :class:`~digital_twin.risk_calculator.RiskCalculator`.

        Returns
        -------
        np.ndarray
            A risk-adjusted row-stochastic 5×5 transition matrix.

        Notes
        -----
        The mapping from *risk_score* to a multiplicative factor is linear:

        - score 0   → multiplier 0.5  (half the baseline progression rate)
        - score 50  → multiplier 1.0  (baseline)
        - score 100 → multiplier 2.0  (double the baseline progression rate)

        After scaling, forward probabilities are clamped to [0, 0.95] so the
        self-loop never drops below 5 %.
        """

        clamped =max (0.0 ,min (100.0 ,risk_score ))
        multiplier =0.5 +(clamped /100.0 )*1.5 

        adjusted =np .copy (self ._base_matrix )
        for state in range (_N_STATES -1 ):
            raw_forward =self ._base_forward [state ]*multiplier 

            forward_prob =min (raw_forward ,0.95 )
            adjusted [state ,state +1 ]=forward_prob 
            adjusted [state ,state ]=1.0 -forward_prob 

        return adjusted 


    def simulate_progression (
    self ,
    current_grade :int ,
    risk_score :float ,
    years :int =5 ,
    n_simulations :int =1000 ,
    )->dict :
        """Run a Monte Carlo simulation of DR progression.

        Parameters
        ----------
        current_grade:
            The patient's current DR grade (0-4).
        risk_score:
            Composite risk score (0-100).
        years:
            Number of years to simulate forward.
        n_simulations:
            Number of independent simulation runs for the Monte Carlo
            estimate.

        Returns
        -------
        dict
            ``current_grade``           – echoed input grade.
            ``yearly_probabilities``    – ``{year: {grade: probability}}``
            ``expected_grade_per_year`` – list of expected grade values.
            ``worst_case_grade``        – 95th-percentile grade at final year.
            ``probability_of_progression`` – P(grade increases within *years*).
            ``median_time_to_progression`` – median year of first progression
            (``None`` if fewer than 50 % of runs progress).
            ``simulations``             – raw ``(n_simulations, years)`` array.

        Raises
        ------
        ValueError
            If *current_grade* is not in 0-4.
        """
        if current_grade <0 or current_grade >4 :
            raise ValueError (
            f"current_grade must be 0-4, got {current_grade}"
            )

        matrix =self ._build_risk_adjusted_matrix (risk_score )
        rng =np .random .default_rng ()


        simulations =np .zeros ((n_simulations ,years ),dtype =np .int32 )

        for sim_idx in range (n_simulations ):
            state =current_grade 
            for yr in range (years ):
                transition_probs =matrix [state ]
                state =rng .choice (_N_STATES ,p =transition_probs )
                simulations [sim_idx ,yr ]=state 


        yearly_probabilities :dict [int ,dict [int ,float ]]={}
        expected_grade_per_year :list [float ]=[]

        for yr in range (years ):
            year_label =yr +1 
            counts =np .bincount (simulations [:,yr ],minlength =_N_STATES )
            probs =counts /n_simulations 
            yearly_probabilities [year_label ]={
            grade :round (float (probs [grade ]),4 )for grade in range (_N_STATES )
            }
            expected_grade_per_year .append (
            round (float (np .mean (simulations [:,yr ])),4 )
            )


        worst_case_grade =int (np .percentile (simulations [:,-1 ],95 ))


        progressed_mask =np .any (simulations >current_grade ,axis =1 )
        probability_of_progression =float (np .mean (progressed_mask ))


        first_progression_years :list [float ]=[]
        for sim_idx in range (n_simulations ):
            for yr in range (years ):
                if simulations [sim_idx ,yr ]>current_grade :
                    first_progression_years .append (yr +1 )
                    break 

        if len (first_progression_years )>=n_simulations *0.5 :
            median_time_to_progression :float |None =float (
            np .median (first_progression_years )
            )
        else :
            median_time_to_progression =None 

        logger .info (
        "Progression simulation complete: grade %d, risk %.1f, "
        "%d sims × %d yrs → P(progression)=%.2f",
        current_grade ,
        risk_score ,
        n_simulations ,
        years ,
        probability_of_progression ,
        )

        return {
        "current_grade":current_grade ,
        "yearly_probabilities":yearly_probabilities ,
        "expected_grade_per_year":expected_grade_per_year ,
        "worst_case_grade":worst_case_grade ,
        "probability_of_progression":round (probability_of_progression ,4 ),
        "median_time_to_progression":(
        round (median_time_to_progression ,2 )
        if median_time_to_progression is not None 
        else None 
        ),
        "simulations":simulations ,
        }


    def get_transition_matrix (self ,risk_score :float )->np .ndarray :
        """Return the risk-adjusted transition matrix (for visualisation).

        Parameters
        ----------
        risk_score:
            Composite risk score (0-100).

        Returns
        -------
        np.ndarray
            5×5 row-stochastic transition matrix.
        """
        return self ._build_risk_adjusted_matrix (risk_score )
