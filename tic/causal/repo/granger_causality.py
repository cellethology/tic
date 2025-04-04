"""
Module: tic.causal.repo.granger_causality

Implements Granger causality for time-series data using
statsmodels.tsa.stattools.grangercausalitytests and, optionally,
a VAR-based automatic lag selection for multivariate analysis.
"""

from typing import Any, Dict, Optional
from statsmodels.tsa.stattools import grangercausalitytests
from tic.causal.base import BaseCausalMethod
from tic.causal.causal_input import CausalInput


class GrangerCausalityMethod(BaseCausalMethod):
    """
    Implements Granger causality for time-series data.

    Mathematical Principle:
        For each lag j (1 to L), the null hypothesis is that the past values
        of X do not help predict Y beyond what is explained by past values of Y
        alone. The method tests this hypothesis using an F-test (SSR F-test). The
        raw p-values are then adjusted using a Bonferroni correction by multiplying
        by L and capping at 1.

    Multivariate Analysis with Auto Lag Selection:
        Instead of specifying a fixed maximum lag, setting `auto_lag=True` will
        automatically select the optimal lag order based on a VAR model fitted to
        the data using the AIC criterion.
    """

    def __init__(
        self,
        name: str = "granger_causality",
        maxlag: Optional[int] = 2,
        auto_lag: bool = False
    ) -> None:
        """
        Initialize the GrangerCausalityMethod.

        Parameters
        ----------
        name : str, optional
            Name of the causal method, by default "granger_causality".
        maxlag : Optional[int], optional
            Maximum number of lags to test (ignored if auto_lag is True),
            by default 2.
        auto_lag : bool, optional
            If True, automatically select the optimal lag number based on a
            VAR model and AIC; by default False.
        """
        super().__init__(name)
        self.maxlag = maxlag
        self.auto_lag = auto_lag
        self._fitted = False

    def fit(self, input_data: CausalInput, *args, **kwargs) -> None:
        """
        Prepare the method on the dataset.

        For Granger causality, no training is required; this method can be used
        for basic data validation.

        Parameters
        ----------
        input_data : CausalInput
            The input data.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.
        """
        self._fitted = True

    def estimate_effect(self, input_data: CausalInput, *args, **kwargs) -> Dict[str, Any]:
        """
        Estimate the causal effect using Granger causality tests.

        Parameters
        ----------
        input_data : CausalInput
            The input causal data.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
                - estimated_effect: None (Granger causality does not yield a single effect).
                - raw_pvalues: Raw p-values for each lag.
                - adjusted_pvalues: Bonferroni-adjusted p-values.
                - best_lag: The lag with the lowest adjusted p-value.
                - p_value: The best adjusted p-value.
                - detailed_stats: Detailed statistics for each lag.

        Raises
        ------
        RuntimeError
            If fit() has not been called before estimate_effect().
        """
        if not self._fitted:
            raise RuntimeError(
                "Must call `fit` before `estimate_effect` for GrangerCausalityMethod."
            )

        treatment_col = input_data.treatment_col
        outcome_col = input_data.outcome_col

        # Extract the treatment and outcome columns, dropping rows with NaNs.
        df = input_data.data[[treatment_col, outcome_col]].dropna()
        if df.empty:
            return {
                "estimated_effect": None,
                "raw_pvalues": {},
                "adjusted_pvalues": {},
                "best_lag": None,
                "p_value": 1.0,
                "detailed_stats": {}
            }

        # Rename columns for convenience.
        df.columns = ["cause", "effect"]

        # Determine the optimal lag order.
        if self.auto_lag:
            from statsmodels.tsa.api import VAR
            # Set an upper bound for lag selection; for example, at most 10 or based on data length.
            maxlags_upper_bound = max(1, min(10, len(df) // 5))
            var_model = VAR(df)
            lag_order_results = var_model.select_order(maxlags_upper_bound)
            # Select the lag order based on the AIC criterion.
            optimal_lag = lag_order_results.aic
            # Ensure at least one lag is used.
            if optimal_lag is None or optimal_lag < 1:
                optimal_lag = 1
        else:
            if self.maxlag is None or self.maxlag < 1:
                raise ValueError("maxlag must be a positive integer if auto_lag is False.")
            optimal_lag = self.maxlag

        # Run Granger causality tests for lags 1 to optimal_lag.
        results = grangercausalitytests(df, maxlag=optimal_lag, verbose=False)

        stats_info: Dict[int, Dict[str, Any]] = {}
        raw_pvalues: Dict[int, float] = {}
        adjusted_pvalues: Dict[int, float] = {}
        best_adj_pvalue = 1.0
        best_lag = None

        for lag in sorted(results.keys()):
            test_dict, _ = results[lag]
            # Retrieve the SSR F-test results: (F_stat, p_value, df_denom, df_num)
            f_stat, p_val, df_denom, df_num = test_dict["ssr_ftest"]

            # Apply Bonferroni correction: multiply by the number of lags tested and cap at 1.
            adj_p_val = min(p_val * optimal_lag, 1.0)

            stats_info[lag] = {
                "f_stat": f_stat,
                "raw_p_value": p_val,
                "adjusted_p_value": adj_p_val,
                "df_denom": df_denom,
                "df_num": df_num
            }
            raw_pvalues[lag] = p_val
            adjusted_pvalues[lag] = adj_p_val

            if adj_p_val < best_adj_pvalue:
                best_adj_pvalue = adj_p_val
                best_lag = lag

        if best_lag is None and raw_pvalues:
            best_lag = min(raw_pvalues.keys())

        return {
            "estimated_effect": None,  # Granger causality does not yield a single numeric effect.
            "raw_pvalues": raw_pvalues,
            "adjusted_pvalues": adjusted_pvalues,
            "best_lag": best_lag,
            "p_value": best_adj_pvalue,
            "detailed_stats": stats_info
        }