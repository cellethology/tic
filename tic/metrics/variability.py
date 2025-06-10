# ==============================================================
# File: tic/metrics/variability.py
# --------------------------------------------------------------
"""Variability metrics for spatial -omics datasets.

This module implements dataset-aware variability measures for:

* Xenium (spatial transcriptomics) → Coefficient of Variation (CV), Shannon entropy(measure of uncertainty)
* Codex  (spatial proteomics)      → Mean-Squared Expression (MSE),
                                         Kurtosis, Skewness, Gini coefficient

Each metric is exposed via a dedicated function, and the public helper
:func:`compute_variability` provides a one-shot API that dispatches to
the appropriate metrics based on ``dataset_type``. See api.py for more details.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from scipy.stats import kurtosis, skew
from scipy.stats import entropy as scipy_entropy 

__all__ = [
    "compute_cv",
    "compute_mse",
    "compute_kurtosis",
    "compute_skewness",
    "compute_gini",
    "compute_variability",
]

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _to_dense(X):
    """Convert sparse or array-like data to a dense ``numpy.ndarray`` (copy-safe)."""
    return X.toarray() if issparse(X) else np.asarray(X)


def _gini(x: np.ndarray) -> float:
    """Compute the Gini coefficient for a 1-D non-negative array.

    Notes
    -----
    * If the input contains negative numbers, the array is shifted so the minimum
      becomes zero (as in *Cowell & Flachaire, 2017*).
    * When the total sum is zero, the function returns ``0.0`` (undefined /
      degenerate case).
    """
    if x.ndim != 1:
        raise ValueError("_gini expects a 1-D array")
    if np.amin(x) < 0:
        x = x - np.amin(x)
    # All zeros → Gini undefined (set to 0)
    if not np.any(x):
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cumx = np.cumsum(x_sorted, dtype=float)
    gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return gini


# -----------------------------------------------------------------------------
# Metric implementations
# -----------------------------------------------------------------------------

def compute_cv(adata: AnnData, *, layer: Optional[str] = None) -> pd.Series:
    """Coefficient of Variation for each gene.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    layer : str, optional
        If given, use ``adata.layers[layer]`` instead of ``adata.X``.

    Returns
    -------
    pandas.Series
        ``index`` = gene names, ``values`` = CV.
    """
    X = _to_dense(adata.layers[layer] if layer else adata.X)
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    # Avoid division by zero → NaN
    cv = std / np.where(mean == 0, np.nan, mean)
    return pd.Series(cv, index=adata.var_names, name="cv")

def compute_entropy(adata: AnnData, *, layer: Optional[str] = None) -> pd.Series:
    """Shannon entropy per gene (across cells).

    For each gene, the expression is normalized to a probability distribution
    (sum=1 across all cells), and Shannon entropy is computed.

    Notes
    -----
    * Input is converted to non-negative values before computing probabilities.
    * Genes with all-zero expression will have entropy 0 (undefined case).

    Returns
    -------
    pandas.Series
        ``index`` = gene names, ``values`` = entropy.
    """
    X = _to_dense(adata.layers[layer] if layer else adata.X)
    X = np.clip(X, a_min=0, a_max=None)  # ensure non-negative
    prob_dist = X / (X.sum(axis=0, keepdims=True) + 1e-12)  # column-wise normalize
    ent = scipy_entropy(prob_dist, axis=0, base=2)
    return pd.Series(ent, index=adata.var_names, name="entropy")

def compute_nonzero_ratio(adata: AnnData, *, layer: Optional[str] = None) -> pd.Series:
    """Compute the ratio of non-zero values for each gene across all cells.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    layer : str, optional
        If given, use ``adata.layers[layer]`` instead of ``adata.X``.

    Returns
    -------
    pandas.Series
        ``index`` = gene names, ``values`` = ratio of non-zero values per gene.
    """
    X = _to_dense(adata.layers[layer] if layer else adata.X)
    ratio = (X != 0).sum(axis=0) / X.shape[0]
    return pd.Series(ratio, index=adata.var_names, name="nonzero_ratio")

def compute_pseudo_nonzero_ratios(
    adata: AnnData,
    *,
    thresholds: Sequence[float] = (0.0, 0.5, 1.0),
    layer: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute pseudo non-zero ratios for z-scored data using various thresholds.

    For each gene and each threshold τ, compute the fraction of cells where
    the z-scored expression exceeds τ.

    Parameters
    ----------
    adata : AnnData
        AnnData object with z-scored expression data.
    thresholds : Sequence[float], default = (0.0, 0.5, 1.0)
        Thresholds above which expression is considered pseudo-non-zero.
    layer : str, optional
        If provided, use `adata.layers[layer]` instead of `adata.X`.

    Returns
    -------
    pandas.DataFrame
        ``index`` = gene names, ``columns`` = pseudo-nonzero ratios for each threshold.
    """
    X = _to_dense(adata.layers[layer] if layer else adata.X)
    results = {}
    for tau in thresholds:
        mask = (X > tau)
        ratios = mask.sum(axis=0) / X.shape[0]
        results[f"pseudo_nonzero>τ={tau}"] = ratios
    return pd.DataFrame(results, index=adata.var_names)

def compute_mse(adata: AnnData, *, layer: Optional[str] = None) -> pd.Series:
    """Mean-Squared Expression for each gene (z-score space).

    Suitable for datasets already scaled (mean≈0, std≈1). Higher MSE implies
    broader expression across cells.
    """
    X = _to_dense(adata.layers[layer] if layer else adata.X)
    mse = np.mean(np.square(X), axis=0)
    return pd.Series(mse, index=adata.var_names, name="mse")


def compute_kurtosis(adata: AnnData, *, layer: Optional[str] = None) -> pd.Series:
    """Excess kurtosis (Fisher definition) per gene."""
    X = _to_dense(adata.layers[layer] if layer else adata.X)
    k = kurtosis(X, axis=0, bias=False, fisher=True, nan_policy="omit")
    return pd.Series(k, index=adata.var_names, name="kurtosis")


def compute_skewness(adata: AnnData, *, layer: Optional[str] = None) -> pd.Series:
    """Skewness per gene."""
    X = _to_dense(adata.layers[layer] if layer else adata.X)
    s = skew(X, axis=0, bias=False, nan_policy="omit")
    return pd.Series(s, index=adata.var_names, name="skewness")


def compute_gini(adata: AnnData, *, layer: Optional[str] = None) -> pd.Series:
    """Gini coefficient per gene.

    Higher Gini indicates expression concentrated in fewer cells (sparse/marker
    gene behaviour).
    """
    X = _to_dense(adata.layers[layer] if layer else adata.X)
    gini_vals = np.apply_along_axis(_gini, 0, X)
    return pd.Series(gini_vals, index=adata.var_names, name="gini")


# Mapping metric names → functions
_METRIC_FUNCS: Dict[str, Callable[[AnnData, Optional[str]], pd.Series]] = {
    "cv": compute_cv,
    "entropy": compute_entropy,
    "nonzero_ratio": compute_nonzero_ratio,
    "pseudo_nonzero_ratios": compute_pseudo_nonzero_ratios,
    "mse": compute_mse,
    "kurtosis": compute_kurtosis,
    "skewness": compute_skewness,
    "gini": compute_gini,
}

