# instrumental_vars.py

import logging
from typing import List, Optional

import pandas as pd
import numpy as np
from linearmodels import IV2SLS

from core.causal.base import BaseCausalMethod
from core.causal.causal_input import CausalInput


logger = logging.getLogger(__name__)


class InstrumentalVariableMethod(BaseCausalMethod):
    """
    Uses a Two-Stage Least Squares (2SLS) approach for identifying
    causal effects with an instrumental variable.
    """

    def __init__(self):
        super().__init__(name="InstrumentalVariables")
        self.iv_model = None        # Will store the fitted IV2SLS results
        self.is_fit_ = False
        self._endog_name = None     # We'll store the treatment (endogenous) variable name here.

    def fit(
        self,
        input_data: CausalInput,
        instrument_col: str,
        covariates: Optional[List[str]] = None,
        add_constant: bool = True,
        *args, 
        **kwargs
    ):
        """
        Fit an IV model (2SLS).
        
        :param input_data: CausalInput with .data, .outcome_col, .treatment_col
        :param instrument_col: The name of the instrument in `data`
        :param covariates: Optional list of additional exogenous regressors
        :param add_constant: If True, add an intercept in the model
        """
        try:
            logger.info("[IV] Fitting 2SLS model...")

            df = input_data.data.copy()
            outcome_col = input_data.outcome_col
            treatment_col = input_data.treatment_col

            if outcome_col is None:
                raise ValueError("[IV] 'outcome_col' must be specified in CausalInput.")
            if treatment_col not in df.columns:
                raise ValueError(f"[IV] Treatment column '{treatment_col}' not found in DataFrame.")
            if instrument_col not in df.columns:
                raise ValueError(f"[IV] Instrument column '{instrument_col}' not found in DataFrame.")

            # Store the treatment name for reference in estimate_effect()
            self._endog_name = treatment_col

            # Outcome
            y = df[outcome_col]

            # Endogenous regressor
            X_endog = df[[treatment_col]]

            # Optional exogenous covariates
            if covariates:
                X_exog = df[covariates]
            else:
                X_exog = pd.DataFrame(index=df.index)  # empty

            # Instruments
            Z = df[[instrument_col]]

            # Optionally add a constant
            if add_constant:
                X_exog = pd.concat(
                    [pd.Series(1, index=df.index, name='const'), X_exog],
                    axis=1
                )

            # Fit 2SLS
            self.iv_model = IV2SLS(
                dependent=y,
                exog=X_exog,
                endog=X_endog,
                instruments=Z
            ).fit()

            self.is_fit_ = True
            logger.info("[IV] IV model fit completed successfully.")

        except Exception as e:
            logger.error(f"[IV] Error during fit: {e}")
            raise

    def estimate_effect(
        self,
        treatment_col: Optional[str] = None,
        *args,
        **kwargs
    ) -> float:
        """
        Return the estimated coefficient on the treatment variable (endogenous regressor).
        
        :param treatment_col: If None, uses the column name stored from fit().
        :return: Estimated 2SLS effect as a float.
        """
        if not self.is_fit_:
            raise RuntimeError("[IV] Cannot estimate effect: model has not been fit.")

        try:
            # Default to whatever was used in fit() if user didn't specify
            if treatment_col is None:
                treatment_col = self._endog_name

            # Now we simply get the coefficient from the fitted model
            effect = self.iv_model.params[treatment_col]
            logger.info(f"[IV] Estimated 2SLS effect for '{treatment_col}': {effect:.4f}")

            return effect
        except Exception as e:
            logger.error(f"[IV] Error during estimate_effect: {e}")
            raise
    
def main():
    np.random.seed(42)
    n = 300

    # Suppose we have:
    #  - 'Z': instrument (correlated with 'D' but not with the error in outcome)
    #  - 'D': endogenous regressor (treatment)
    #  - outcome 'Y': depends on D but also has an unobserved confounder correlated with D
    #  - exog covariates 'X1', 'X2' that are not correlated with the error term

    Z = np.random.randn(n)  # instrument
    X1 = np.random.randn(n)
    X2 = np.random.randn(n)

    # Let's define the relationship:
    # D = 0.5 * Z + 0.3 * X1 - 0.2 * X2 + u  (u correlated w/ error in Y)
    u = np.random.randn(n)
    D = 0.5 * Z + 0.3 * X1 - 0.2 * X2 + 0.7 * u

    # Y = 2.0 * D + 0.2 * X1 + 0.1 * X2 + e
    # e correlated with u => OLS on D would be biased
    e = 0.5 * u + np.random.randn(n)  # correlated with u => endogeneity
    Y = 2.0 * D + 0.2 * X1 + 0.1 * X2 + e

    df = pd.DataFrame({
        "Z": Z,
        "D": D,
        "X1": X1,
        "X2": X2,
        "Y": Y
    })

    # Build the CausalInput (outcome=Y, treat=D)
    c_input = CausalInput(
        data=df,
        treatment_col="D",
        outcome_col="Y",
        covariates=["X1", "X2"]  # exogenous regressors
    )

    # Instantiate the IV method
    iv_method = InstrumentalVariableMethod()

    # Fit the model
    iv_method.fit(
        input_data=c_input,
        instrument_col="Z",
        covariates=["X1", "X2"],
        add_constant=True
    )

    # Estimate the effect of D on Y
    effect_estimate = iv_method.estimate_effect()
    print("Estimated 2SLS effect of D on Y:", effect_estimate)

if __name__ == "__main__":
    main()