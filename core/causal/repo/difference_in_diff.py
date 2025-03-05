# difference_in_diff.py

import logging
from typing import Optional, List

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from core.causal.base import BaseCausalMethod
from core.causal.causal_input import CausalInput

logger = logging.getLogger(__name__)

class DifferenceInDifferencesMethod(BaseCausalMethod):
    """
    Implements a simple 2x2 Difference-in-Differences (DiD) approach:
      outcome ~ group_col + time_col + group_col:time_col (+ optional covariates)

    The coefficient on group_col:time_col is the DiD estimate.
    """

    def __init__(self):
        super().__init__(name="DifferenceInDifferences")
        self.did_model = None
        self.is_fit_ = False
        self.group_col_ = None
        self.time_col_ = None
        self.outcome_col_ = None

    def fit(
        self,
        input_data: CausalInput,
        group_col: str,
        time_col: str,
        covariates: Optional[List[str]] = None,
        add_constant: bool = True,
        *args,
        **kwargs
    ):
        """
        :param input_data: CausalInput with:
            - data (pd.DataFrame)
            - outcome_col (str)
        :param group_col: Name of binary column indicating group (0=control, 1=treated).
        :param time_col: Name of binary column indicating time (0=pre, 1=post).
        :param covariates: Optional list of additional columns to control for.
        :param add_constant: If True, statsmodels will automatically add an intercept
                             if not already included in the formula.
        """
        try:
            logger.info("[DiD] Fitting difference-in-differences model.")
            df = input_data.data.copy()
            self.outcome_col_ = input_data.outcome_col
            if self.outcome_col_ is None:
                raise ValueError("[DiD] The outcome_col must be set in CausalInput.")

            # Validate columns
            if group_col not in df.columns:
                raise ValueError(f"[DiD] group_col '{group_col}' not in DataFrame.")
            if time_col not in df.columns:
                raise ValueError(f"[DiD] time_col '{time_col}' not in DataFrame.")

            self.group_col_ = group_col
            self.time_col_ = time_col

            # Build formula: outcome ~ group + time + group:time (+ covariates)
            # e.g. "Y ~ group_col * time_col + X1 + X2"
            # The interaction (group_col:time_col) is the DiD coefficient.

            # Base formula with interaction
            formula = f"{self.outcome_col_} ~ {group_col} * {time_col}"

            # If user has extra covariates, add them
            if covariates:
                covariate_terms = " + ".join(covariates)
                # Expand formula to include those covariates plus the existing interaction
                # The formula for interactions won't automatically cover covariates, so we do:
                formula = f"{formula} + {covariate_terms}"

            # Optionally, remove the intercept if you prefer (add_constant=False).
            # By default, statsmodels formula includes an intercept.
            # If add_constant=False, we can do: formula += " - 1"
            if not add_constant:
                formula += " - 1"

            logger.info(f"[DiD] Using formula: {formula}")
            self.did_model = smf.ols(formula=formula, data=df).fit()
            self.is_fit_ = True

            logger.info("[DiD] DiD model fit completed.")
        except Exception as e:
            logger.error(f"[DiD] Error during fit: {e}")
            raise

    def estimate_effect(self, *args, **kwargs) -> float:
        """
        Returns the estimated DiD effect, i.e., the coefficient on group_col:time_col
        from the fitted OLS model.
        
        :return: float, the DiD estimate (interaction coefficient).
        """
        if not self.is_fit_ or self.did_model is None:
            raise RuntimeError("[DiD] Cannot estimate effect: model has not been fit.")

        try:
            interaction_term = f"{self.group_col_}:{self.time_col_}"
            if interaction_term not in self.did_model.params:
                raise ValueError(f"[DiD] Interaction term '{interaction_term}' was not found in the model parameters.")

            did_estimate = self.did_model.params[interaction_term]

            logger.info(f"[DiD] Estimated DiD effect (interaction) = {did_estimate:.4f}")
            return did_estimate
        except Exception as e:
            logger.error(f"[DiD] Error during estimate_effect: {e}")
            raise

def main():
    np.random.seed(42)
    n = 400
    baseline = 5.0
    effect_of_treated = 0.5
    effect_of_post = -1.0
    did_effect = 2.0  # the "interaction" effect we want to recover

    # We create group and time arrays
    group = np.random.binomial(1, 0.5, size=n)  # 0 or 1
    time = np.random.binomial(1, 0.5, size=n)   # 0 or 1

    # Construct outcome
    Y = (baseline
         + effect_of_treated * group
         + effect_of_post * time
         + did_effect * group * time
         + np.random.normal(0, 1, size=n))

    df = pd.DataFrame({
        "group": group,  # treated=1, control=0
        "post": time,    # post=1, pre=0
        "Y": Y
    })

    # Build the input object
    c_input = CausalInput(
        data=df,
        treatment_col="group",   # If your pipeline needs it, though DiD doesn't strictly require
        outcome_col="Y",
        covariates=[]
    )

    # Instantiate the method
    did_method = DifferenceInDifferencesMethod()

    # Fit the DiD model
    did_method.fit(
        input_data=c_input,
        group_col="group",
        time_col="post",
        covariates=None,
        add_constant=True
    )

    # Estimate the effect (coefficient on group:post)
    effect_est = did_method.estimate_effect()
    print("Estimated DiD effect (interaction coefficient):", effect_est)

if __name__ == "__main__":
    main()