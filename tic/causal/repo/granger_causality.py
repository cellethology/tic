from statsmodels.tsa.stattools import grangercausalitytests
from typing import Any, Dict

from tic.causal.base import BaseCausalMethod
from tic.causal.causal_input import CausalInput

class GrangerCausalityMethod(BaseCausalMethod):
    """
    A causal method implementing Granger causality for time-series data,
    using statsmodels.tsa.stattools.grangercausalitytests.
    """
    def __init__(self, name: str = "granger_causality", maxlag: int = 2):
        super().__init__(name)
        self.maxlag = maxlag
        self._fitted = False

    def fit(self, input_data: CausalInput, *args, **kwargs):
        # No training is typically necessary, but you might do data checks here.
        self._fitted = True

    def estimate_effect(self, input_data: CausalInput, *args, **kwargs) -> Dict[str, Any]:
        if not self._fitted:
            raise RuntimeError(
                "Must call `fit` before `estimate_effect` for GrangerCausalityMethod."
            )

        treatment_col = input_data.treatment_col
        outcome_col = input_data.outcome_col
        
        # Extract just these two columns, dropping NaNs if needed
        df = input_data.data[[treatment_col, outcome_col]].dropna()

        if df.empty:
            return {
                "estimated_effect": None,
                "p_value": 1.0,
                "best_lag": None,
                "detailed_stats": {}
            }

        # Rename columns to standard [cause, effect] for convenience
        df.columns = ["cause", "effect"]

        # Perform Granger causality tests up to `self.maxlag`
        # results[lag] is a dict with test results keyed by "ssr_ftest", "lrtest", etc.
        results = grangercausalitytests(df, maxlag=self.maxlag, verbose=False)

        best_pvalue = 1.0
        best_lag = None
        stats_info = {}

        for lag in results:
            # Each value is (test_dict, extra_list/array)
            test_dict, extra_list = results[lag]

            ssr_ftest_tuple = test_dict["ssr_ftest"]  # e.g. (F_stat, p_value, df_denom, df_num)
            test_stat, p_val, df_denom, df_num = ssr_ftest_tuple

            print(f"Lag {lag} => F={test_stat}, p={p_val}, df_denom={df_denom}, df_num={df_num}")
            
            stats_info[lag] = {
                "f_stat": test_stat,
                "p_value": p_val,
                "df_denom": df_denom,
                "df_num": df_num
            }
            # Keep track of minimal p-value
            if p_val < best_pvalue:
                best_pvalue = p_val
                best_lag = lag

        return {
            "estimated_effect": None,   # Granger doesn't provide a single numeric ATE
            "p_value": best_pvalue,
            "best_lag": best_lag,
            "detailed_stats": stats_info
        }