"""
Public-facing API for metrics module.

Provides single and batch processing for monotonicity/trend analysis,
with optional sorting and plotting.
"""

from typing import List, Optional, Tuple, Literal
import numpy as np

from .monotonicity import spearman_monotonicity, kendall_monotonicity
from .trend import mann_kendall_trend, linear_regression_trend


def calculate_monotonicity(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    method: Literal["spearman", "kendall"] = "spearman"
) -> dict:
    '''
    Calculate the monotonicity of a time series.

    Parameters
    ----------
    y : np.ndarray
        The time series to analyze.
    x : np.ndarray, optional
        The time points corresponding to the time series. If not provided, the indices of the time series will be used.
    method : str, optional
        The method to use for calculating the monotonicity. Can be "spearman" or "kendall".

    Returns
    -------
    dict:
        method: str
        correlation: float
        p_value: float
    '''
    if method == "spearman":
        return spearman_monotonicity(y, x)
    elif method == "kendall":
        return kendall_monotonicity(y, x)
    raise ValueError(f"Unsupported monotonicity method: {method}")


def calculate_trend(
    y: np.ndarray,
    x: Optional[np.ndarray] = None,
    method: Literal["mann_kendall", "linear_regression"] = "mann_kendall"
) -> dict:
    '''
    Calculate the trend of a time series.

    Parameters
    ----------
    y : np.ndarray
        The time series to analyze.
    x : np.ndarray, optional
        The time points corresponding to the time series. If not provided, the indices of the time series will be used.
    method : str, optional
        The method to use for calculating the trend. Can be "mann_kendall" or "linear_regression".

    Returns
    -------
    dict: No regular format for the output.
    '''
    if method == "mann_kendall":
        return mann_kendall_trend(y, x)
    elif method == "linear_regression":
        return linear_regression_trend(y, x)
    raise ValueError(f"Unsupported trend method: {method}")


def rank_by_monotonicity(
    time_series_list: List[np.ndarray],
    method: Literal["spearman", "kendall"] = "spearman",
    sort_descending: bool = True,
    return_scores: bool = False
) -> List[int]:
    """
    Ranks multiple time series by their monotonicity strength.

    Parameters
    ----------
    time_series_list : List[np.ndarray]
        List of 1D time series arrays.
    method : str
        Monotonicity method.
    sort_descending : bool
        If True, sort from highest to lowest.
    return_scores : bool
        If True, also return the score list.

    Returns
    -------
    List[int] or (List[int], List[float])
        Sorted indices, optionally with scores.
    """
    scores = []
    for series in time_series_list:
        result = calculate_monotonicity(series, method=method)
        scores.append(abs(result["correlation"]))

    indices = np.argsort(scores)[::-1] if sort_descending else np.argsort(scores)
    if return_scores:
        return indices.tolist(), [scores[i] for i in indices]
    return indices.tolist()


def rank_by_trend(
    time_series_list: List[np.ndarray],
    method: Literal["mann_kendall", "linear_regression"] = "mann_kendall",
    sort_descending: bool = True,
    return_scores: bool = False
) -> List[int]:
    """
    Ranks multiple time series by trend strength.

    Returns
    -------
    List[int] or (List[int], List[float])
        Sorted indices, optionally with scores.
    """
    scores = []
    for series in time_series_list:
        result = calculate_trend(series, method=method)
        if method == "mann_kendall":
            score = abs(result["statistic"])
        elif method == "linear_regression":
            score = abs(result["slope"])
        scores.append(score)

    indices = np.argsort(scores)[::-1] if sort_descending else np.argsort(scores)
    if return_scores:
        return indices.tolist(), [scores[i] for i in indices]
    return indices.tolist()

