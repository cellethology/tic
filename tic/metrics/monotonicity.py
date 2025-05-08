'''
This module implements the monotonicity metric.

Current support:

    - Spearman's rank correlation coefficient
'''
"""
Implements functions to compute monotonicity measures, such as
Spearman and Kendall correlation for a time series.
"""

from typing import Optional, Union
import numpy as np
from scipy.stats import spearmanr, kendalltau


def _prepare_data(
    y: np.ndarray,
    x: Optional[np.ndarray] = None
) -> Union[np.ndarray, np.ndarray]:
    """
    Internal helper function to prepare x and y data for monotonicity metrics.

    Parameters
    ----------
    y : np.ndarray
        Array of measurements.
    x : np.ndarray, optional
        Array of times. If None, it will be automatically created.

    Returns
    -------
    x : np.ndarray
        Processed time array.
    y : np.ndarray
        Original measurements (unmodified).
    """
    if x is None:
        x = np.arange(len(y))
    else:
        if len(x) != len(y):
            raise ValueError("The length of x must match the length of y.")
    return x, y


def spearman_monotonicity(
    y: np.ndarray,
    x: Optional[np.ndarray] = None
) -> dict:
    """
    Computes the Spearman rank correlation coefficient between x and y,
    a common measure for monotonic relationship.

    Parameters
    ----------
    y : np.ndarray
        Measurements in chronological order.
    x : np.ndarray, optional
        Time array. If None, will default to np.arange(len(y)).

    Returns
    -------
    dict
        A dictionary containing 'correlation' and 'p_value'.
    """
    x, y = _prepare_data(y, x)
    corr, p_val = spearmanr(x, y)
    return {
        "method": "spearman",
        "correlation": corr,
        "p_value": p_val
    }


def kendall_monotonicity(
    y: np.ndarray,
    x: Optional[np.ndarray] = None
) -> dict:
    """
    Computes the Kendall tau rank correlation coefficient between x and y,
    another robust measure for monotonic relationship.

    Parameters
    ----------
    y : np.ndarray
        Measurements in chronological order.
    x : np.ndarray, optional
        Time array. If None, will default to np.arange(len(y)).

    Returns
    -------
    dict
        A dictionary containing 'correlation' and 'p_value'.
    """
    x, y = _prepare_data(y, x)
    tau, p_val = kendalltau(x, y)
    return {
        "method": "kendall",
        "correlation": tau,
        "p_value": p_val
    }