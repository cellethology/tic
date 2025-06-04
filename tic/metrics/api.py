"""
Public-facing API for metrics module.

Provides single and batch processing for monotonicity/trend analysis,
with optional sorting and plotting.
"""

from typing import List, Optional, Sequence, Tuple, Literal
from anndata import AnnData
import numpy as np
import pandas as pd


from .monotonicity import spearman_monotonicity, kendall_monotonicity
from .trend import mann_kendall_trend, linear_regression_trend
from .variability import _METRIC_FUNCS

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

def compute_variability(
    adata: AnnData,
    *,
    dataset_type: Literal["xenium", "codex"],
    layer: Optional[str] = None,
    metrics: Optional[Sequence[str]] = None,
    drop_na: bool = True,
) -> pd.DataFrame:
    """Compute variability metrics tailored to the given ``dataset_type``.

    Parameters
    ----------
    adata : AnnData
        Expression matrix (cells × genes).
    dataset_type : {'xenium', 'codex'}
        Determines which metrics are computed by default:
        * xenium → ['cv']
        * codex  → ['mse', 'kurtosis', 'skewness', 'gini']
    layer : str, optional
        Anndata layer to use (default ``None`` → ``adata.X``).
    metrics : Sequence[str], optional
        Manually specify metrics (subset of CV/MSE/Kurtosis/Skewness/Gini).
    drop_na : bool, default ``True``
        Remove genes with NaN values (e.g. CV when mean==0).

    Returns
    -------
    pandas.DataFrame
        ``index`` = gene names, each column = metric.
    """

    if metrics is None:
        metrics = (
            ["cv"] if dataset_type == "xenium" else ["mse", "kurtosis", "skewness", "gini"]
        )

    invalid = set(metrics) - _METRIC_FUNCS.keys()
    if invalid:
        raise ValueError(f"Unsupported metric(s): {sorted(invalid)}")

    # Compute selected metrics and concatenate
    results = [
        _METRIC_FUNCS[m](adata, layer=layer).rename(m)  # type: ignore[arg-type]
        for m in metrics
    ]
    df = pd.concat(results, axis=1)

    # Optional NA pruning
    if drop_na:
        df = df.dropna(axis=0, how="any")

    return df