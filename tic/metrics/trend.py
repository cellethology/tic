"""
This module implements the trend metric.

Implements functions to detect trends in time series data,
such as Mann-Kendall or linear regression slope tests.
"""

from typing import Optional, Union
import numpy as np
from scipy.stats import linregress, norm


def _prepare_data(
    y: np.ndarray,
    x: Optional[np.ndarray] = None
) -> Union[np.ndarray, np.ndarray]:
    """Internal helper function to prepare x and y for trend analysis."""
    if x is None:
        x = np.arange(len(y))
    else:
        if len(x) != len(y):
            raise ValueError("The length of x must match the length of y.")
    return x, y


def mann_kendall_trend(
    y: np.ndarray,
    x: Optional[np.ndarray] = None
) -> dict:
    """
    Perform the Mann-Kendall test to detect a monotonic trend in a time series.

    Parameters
    ----------
    y : np.ndarray
        Measurements in chronological order.
    x : np.ndarray, optional
        Time array. If None, will default to np.arange(len(y)).

    Returns
    -------
    dict
        A dictionary containing the test statistic and p-value.
        Example:
        {
            'method': 'mann_kendall',
            'trend': 'increasing'/'decreasing'/'no_trend',
            'statistic': float,
            'p_value': float
        }
    """
    _, y = _prepare_data(y, x)
    n = len(y)

    # Compute S statistic
    s = 0
    for k in range(n - 1):
        s += np.sum(np.sign(y[k + 1:] - y[k]))

    # Tie correction: count repeated values
    unique, counts = np.unique(y, return_counts=True)
    ties = counts[counts > 1]

    # Compute variance of S
    var_s = (
        n * (n - 1) * (2 * n + 5) -
        np.sum(ties * (ties - 1) * (2 * ties + 5))
    ) / 18.0

    # Compute Z statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    # Compute p-value from normal distribution
    p_value = 2 * (1 - norm.cdf(abs(z)))

    # Determine trend direction
    if p_value < 0.05:
        trend = "increasing" if z > 0 else "decreasing"
    else:
        trend = "no_trend"

    return {
        "method": "mann_kendall",
        "trend": trend,
        "statistic": z,
        "p_value": p_value
    }


def linear_regression_trend(
    y: np.ndarray,
    x: Optional[np.ndarray] = None
) -> dict:
    """
    Fits a simple linear regression of y on x and reports the slope and p-value.

    Parameters
    ----------
    y : np.ndarray
        Measurements in chronological order.
    x : np.ndarray, optional
        Time array. If None, will default to np.arange(len(y)).

    Returns
    -------
    dict
        A dictionary containing slope, intercept, and p-value of the slope.
    """
    x, y = _prepare_data(y, x)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    return {
        "method": "linear_regression",
        "slope": slope,
        "intercept": intercept,
        "p_value": p_value,
        "r_value": r_value,
        "std_err": std_err
    }