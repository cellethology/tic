#!/usr/bin/env python3

import pytest
import numpy as np
import pandas as pd

from tic.causal.causal_input import CausalInput
from tic.causal.repo.granger_causality import GrangerCausalityMethod

# ----------------- Tests for Granger causality  Module -----------------

@pytest.fixture
def mock_causal_input():
    """
    Create a synthetic time-series dataset and return a corresponding CausalInput.
    The dataset contains:
      - 'X': treatment time series (random noise).
      - 'Y': outcome time series (lagged treatment with added noise).
    """
    np.random.seed(42)
    n_obs = 50
    treatment = np.random.randn(n_obs)
    outcome = np.roll(treatment, 1) + 0.1 * np.random.randn(n_obs)
    df = pd.DataFrame({'X': treatment, 'Y': outcome})
    return CausalInput(data=df, treatment_col='X', outcome_col='Y')


def test_estimate_effect_without_fit_raises_error(mock_causal_input):
    """
    Test that calling estimate_effect without prior fit() raises a RuntimeError.
    """
    method = GrangerCausalityMethod(maxlag=3, auto_lag=False)
    with pytest.raises(RuntimeError):
        method.estimate_effect(mock_causal_input)


def test_estimate_effect_fixed_lag(mock_causal_input):
    """
    Test that the fixed-lag mode returns a result with expected keys and types,
    and that the best lag is computed as the lag with the minimum adjusted p-value.
    """
    method = GrangerCausalityMethod(maxlag=3, auto_lag=False)
    method.fit(mock_causal_input)
    result = method.estimate_effect(mock_causal_input)

    expected_keys = {
        "estimated_effect",
        "raw_pvalues",
        "adjusted_pvalues",
        "best_lag",
        "p_value",
        "detailed_stats"
    }
    assert expected_keys.issubset(result.keys())
    assert result["estimated_effect"] is None
    assert isinstance(result["raw_pvalues"], dict)
    assert isinstance(result["adjusted_pvalues"], dict)

    # If raw p-values are available, compute the expected best lag
    if result["raw_pvalues"]:
        expected_best_lag = min(
            result["raw_pvalues"],
            key=lambda lag: result["adjusted_pvalues"].get(lag, 1.0)
        )
        # The best lag should be an integer (not None)
        assert result["best_lag"] == expected_best_lag, (
            f"Expected best lag {expected_best_lag}, got {result['best_lag']}"
        )
    else:
        assert result["best_lag"] is None

    assert isinstance(result["p_value"], float)
    assert isinstance(result["detailed_stats"], dict)


def test_estimate_effect_auto_lag(mock_causal_input):
    """
    Test that the auto lag mode returns a result with expected keys and types,
    and that the best lag is computed as the lag with the minimum adjusted p-value.
    """
    method = GrangerCausalityMethod(maxlag=3, auto_lag=True)
    method.fit(mock_causal_input)
    result = method.estimate_effect(mock_causal_input)

    expected_keys = {
        "estimated_effect",
        "raw_pvalues",
        "adjusted_pvalues",
        "best_lag",
        "p_value",
        "detailed_stats"
    }
    assert expected_keys.issubset(result.keys())
    assert result["estimated_effect"] is None
    assert isinstance(result["raw_pvalues"], dict)
    assert isinstance(result["adjusted_pvalues"], dict)

    if result["raw_pvalues"]:
        expected_best_lag = min(
            result["raw_pvalues"],
            key=lambda lag: result["adjusted_pvalues"].get(lag, 1.0)
        )
        assert result["best_lag"] == expected_best_lag, (
            f"Expected best lag {expected_best_lag}, got {result['best_lag']}"
        )
    else:
        assert result["best_lag"] is None

    assert isinstance(result["p_value"], float)
    assert isinstance(result["detailed_stats"], dict)