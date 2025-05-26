"""
Module: tic.causal.repo.granger_causality

Implements Granger causality for time-series data using
statsmodels.tsa.stattools.grangercausalitytests and, optionally,
a VAR-based automatic lag selection for multivariate analysis.
"""
from typing import Any, Dict, Optional
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tools.sm_exceptions import InfeasibleTestError

from ..base import BaseCausalMethod
from ..causal_input import CausalInput


class GrangerCausalityMethod(BaseCausalMethod):
    """
    Implements Granger causality for time-series data.

    Utilizes statsmodels.tsa.stattools.grangercausalitytests under the hood,
    with optional automatic lag selection via VAR + AIC criterion.

    Attributes
    ----------
    name : str
        Name identifier for the causal method.
    maxlag : Optional[int]
        Maximum lag order to test (ignored if auto_lag is True).
    auto_lag : bool
        Whether to automatically select lag via VAR AIC.
    """

    def __init__(
        self,
        name: str = "granger_causality",
        maxlag: Optional[int] = 2,
        auto_lag: bool = False,
    ) -> None:
        """
        Initialize the Granger causality method.

        Parameters
        ----------
        name : str, optional
            Method name (default "granger_causality").
        maxlag : Optional[int], optional
            Max number of lags to test if auto_lag is False (default 2).
        auto_lag : bool, optional
            Whether to auto-select lag based on VAR AIC (default False).
        """
        super().__init__(name)
        self.maxlag = maxlag
        self.auto_lag = auto_lag
        self._fitted = False

    def fit(self, input_data: CausalInput, *args, **kwargs) -> None:
        """
        Prepare the method on the dataset.

        For Granger causality no training is required; this sets the fitted flag.

        Parameters
        ----------
        input_data : CausalInput
            The input data container (unused here).
        """
        self._fitted = True

    def estimate_effect(
        self, input_data: CausalInput, *args, **kwargs
    ) -> Dict[str, Any]:
        """
        Estimate causal effect using Granger causality tests.

        Steps
        -----
        1. Drop NaNs and rename series to 'cause' and 'effect'.
        2. Determine optimal lag (auto_lag vs fixed maxlag).
        3. Short‑circuit if either series is constant → return empty result.
        4. Run grangercausalitytests; catch InfeasibleTestError → empty result.
        5. Compile p-values, adjusted p-values, statistics, and best‑lag parameters.

        Parameters
        ----------
        input_data : CausalInput
            Contains `data` DataFrame, `treatment_col`, and `outcome_col`.

        Returns
        -------
        Dict[str, Any]
            A result dictionary with keys:
            - estimated_effect: None
            - raw_pvalues: dict[int, float]
            - adjusted_pvalues: dict[int, float]
            - best_lag: Optional[int]
            - p_value: float
            - detailed_stats: dict[int, dict]
            - best_model_parameters: Optional[list[float]]
        """
        if not self._fitted:
            raise RuntimeError("Must call `fit` before `estimate_effect`.")

        # Prepare data
        df = input_data.data[
            [input_data.treatment_col, input_data.outcome_col]
        ].dropna()
        if df.empty:
            return self._empty_result()
        df.columns = ["cause", "effect"]

        # Select lag order
        if self.auto_lag:
            from statsmodels.tsa.api import VAR

            maxlags = max(1, min(10, len(df) // 5))
            var_model = VAR(df)
            order = var_model.select_order(maxlags)
            optimal_lag = order.aic or 1
            optimal_lag = max(optimal_lag, 1)
        else:
            if not self.maxlag or self.maxlag < 1:
                raise ValueError("`maxlag` must be a positive integer.")
            optimal_lag = self.maxlag

        # Shortcut for constant series
        if df["cause"].nunique() < 2 or df["effect"].nunique() < 2:
            return self._empty_result()

        # Run Granger causality, handle infeasible cases
        try:
            results = grangercausalitytests(df, maxlag=optimal_lag, verbose=False)
        except InfeasibleTestError:
            return self._empty_result()

        return self._compile_results(results, optimal_lag)

    def _compile_results(
        self, results: Dict[int, Any], optimal_lag: int
    ) -> Dict[str, Any]:
        """
        Compile raw granger test outputs into standardized result dict.

        Parameters
        ----------
        results : dict
            Output from grangercausalitytests: lag -> (test_dict, models).
        optimal_lag : int
            Number of lags tested (for Bonferroni correction).

        Returns
        -------
        Dict[str, Any]
            Structured result dictionary.
        """
        raw_pvalues: Dict[int, float] = {}
        adjusted_pvalues: Dict[int, float] = {}
        detailed_stats: Dict[int, Dict[str, Any]] = {}
        best_lag: Optional[int] = None
        best_adj_p = 1.0
        best_params: Optional[list[float]] = None

        for lag, (test_dict, models) in results.items():
            f_stat, p_val, df_denom, df_num = test_dict["ssr_ftest"]
            coeffs = models[1].params.tolist()
            adj_p = min(p_val * optimal_lag, 1.0)

            raw_pvalues[lag] = p_val
            adjusted_pvalues[lag] = adj_p
            detailed_stats[lag] = {
                "f_stat": f_stat,
                "raw_p_value": p_val,
                "adjusted_p_value": adj_p,
                "df_denom": df_denom,
                "df_num": df_num,
                "coefficients": coeffs,
            }

            if adj_p < best_adj_p:
                best_adj_p = adj_p
                best_lag = lag
                best_params = coeffs

        # If no best_lag chosen but tests ran, pick the first lag
        if best_lag is None and raw_pvalues:
            first = min(raw_pvalues.keys())
            best_lag = first
            best_params = detailed_stats[first]["coefficients"]

        return {
            "estimated_effect": None,
            "raw_pvalues": raw_pvalues,
            "adjusted_pvalues": adjusted_pvalues,
            "best_lag": best_lag,
            "p_value": best_adj_p,
            "detailed_stats": detailed_stats,
            "best_model_parameters": best_params,
        }

    def _empty_result(self) -> Dict[str, Any]:
        """
        Return a standardized 'empty' result when test is not feasible.

        All p-values = 1.0, no coefficients.
        """
        return {
            "estimated_effect": None,
            "raw_pvalues": {},
            "adjusted_pvalues": {},
            "best_lag": None,
            "p_value": 1.0,
            "detailed_stats": {},
            "best_model_parameters": None,
        }