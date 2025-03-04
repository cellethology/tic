# propensity_score.py

import logging

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from core.causal.base import BaseCausalMethod
from core.causal.causal_input import CausalInput


logger = logging.getLogger(__name__)


class PropensityScoreMethod(BaseCausalMethod):
    """
    Estimate treatment effects by propensity score weighting or stratification.
    This example demonstrates a simple IPW-based ATE estimation.
    """

    def __init__(self):
        super().__init__(name="PropensityScore")
        self.model = None         # Will store a trained LogisticRegression model
        self.propensity_scores_ = None  # Will store the computed scores
        self.is_fit_ = False

    def fit(
        self, 
        input_data: CausalInput, 
        max_iter: int = 1000,
        solver: str = "lbfgs",
        *args, 
        **kwargs
    ):
        """
        Fit a propensity score model (logistic regression: P(T=1 | X)).

        :param input_data: CausalInput containing:
            - data (pd.DataFrame)
            - treatment_col (str)
            - covariates (list of str)
        :param max_iter: Max iterations for LogisticRegression
        :param solver: Solver used in LogisticRegression
        """
        try:
            logger.info("[PropensityScore] Fitting logistic regression for P(T=1|X).")

            df = input_data.data.copy()
            treat_col = input_data.treatment_col
            covariates = input_data.covariates

            # Basic checks
            if treat_col not in df.columns:
                raise ValueError(f"Treatment column '{treat_col}' not found in data.")
            if not covariates:
                raise ValueError("No covariates provided for propensity score estimation.")

            X = df[covariates]
            y = df[treat_col]

            # Fit logistic regression to get propensity scores
            self.model = LogisticRegression(solver=solver, max_iter=max_iter)
            self.model.fit(X, y)
            
            # Store propensity scores
            self.propensity_scores_ = self.model.predict_proba(X)[:, 1]
            self.is_fit_ = True

            logger.info("[PropensityScore] Successfully fit logistic regression model.")
        except Exception as e:
            logger.error(f"[PropensityScore] Error during fit: {e}")
            raise

    def estimate_effect(
        self, 
        input_data: CausalInput, 
        effect_type: str = "ATE", 
        eps: float = 1e-6,
        *args, 
        **kwargs
    ) -> float:
        """
        Estimate treatment effect using inverse probability weighting (IPW).
        
        :param input_data: CausalInput object with:
            - data (pd.DataFrame)
            - treatment_col (str)
            - outcome_col (str)
        :param effect_type: "ATE" by default. (Other variants like "ATT" would require different weighting.)
        :param eps: Small value to avoid division by zero or extremely small denominators
        :return: Estimated treatment effect (float)
        """
        if not self.is_fit_:
            raise RuntimeError("[PropensityScore] Cannot estimate effect before calling fit().")

        if input_data.outcome_col is None:
            raise ValueError("[PropensityScore] Outcome column must be specified in CausalInput.")

        if effect_type != "ATE":
            logger.warning(f"[PropensityScore] Only 'ATE' is implemented in this example. Received: {effect_type}")

        try:
            df = input_data.data.copy()
            treat_col = input_data.treatment_col
            outcome_col = input_data.outcome_col

            # Retrieve the propensity scores from the fitted model
            ps = self.propensity_scores_
            if ps is None or len(ps) != len(df):
                raise ValueError("[PropensityScore] Propensity scores do not match dataset length.")

            # Safeguard to avoid division by zero
            ps = np.clip(ps, eps, 1 - eps)

            T = df[treat_col].values
            Y = df[outcome_col].values

            # IPW for ATE:
            # Weighted outcome for T=1 is Y / ps, for T=0 is Y / (1-ps).
            # ATE = average over i of [ T_i * Y_i / e_i - (1 - T_i) * Y_i / (1 - e_i ) ]

            weighted_outcomes = T * Y / ps - (1 - T) * Y / (1 - ps)
            ate_ipw = np.mean(weighted_outcomes)

            logger.info(f"[PropensityScore] Estimated ATE (IPW) = {ate_ipw:.4f}")
            return ate_ipw
        except Exception as e:
            logger.error(f"[PropensityScore] Error during estimate_effect: {e}")
            raise
    
def main():
    np.random.seed(42)
    n = 200

    # Create dummy data
    df = pd.DataFrame({
        "treated": np.random.binomial(1, 0.3, size=n),
        "cov1": np.random.randn(n),
        "cov2": np.random.randn(n),
    })
    
    # True outcome mechanism (just a demonstration)
    # Let's say the treatment effect is +1 on average
    df["outcome"] = 2 * df["cov1"] - df["cov2"] + 1.0 * df["treated"] + np.random.randn(n) * 0.5

    # Build the CausalInput
    c_input = CausalInput(
        data=df, 
        treatment_col="treated",
        outcome_col="outcome",
        covariates=["cov1", "cov2"]
    )

    # Instantiate the method
    pscore_method = PropensityScoreMethod()

    # Fit logistic regression (propensity model)
    pscore_method.fit(
        input_data=c_input,
        max_iter=500
    )

    # Estimate the effect via IPW
    ate_estimate = pscore_method.estimate_effect(c_input, effect_type="ATE")
    print("Estimated ATE (IPW):", ate_estimate)

if __name__ == "__main__":
    main()