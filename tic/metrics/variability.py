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

from typing import Callable, Dict, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from scipy.stats import normaltest, skew, kurtosis
from scipy.stats import entropy as scipy_entropy 
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture 
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
    thresholds: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0),
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

def _is_uni_normal(z: np.ndarray,
                   p_normal: float = 0.05,
                   max_abs_skew: float = 0.5,
                   max_kurtosis: float = 3.5) -> bool:
    """
    Determine if a 1-D z-score vector is approximately unimodal normal distribution.

    Conditions (all must be met for True):
    1. Shapiro/D'Agostino normality test p > p_normal
    2. |skew| < max_abs_skew   (symmetric)
    3. kurtosis < max_kurtosis  (not too sharp/not too fat tail)
    """
    z1d = z.ravel()
    if z1d.size < 20:        
        return False
    if normaltest(z1d, nan_policy="omit")[1] < p_normal:
        return False
    if np.abs(skew(z1d, nan_policy="omit")) > max_abs_skew:
        return False
    if kurtosis(z1d, fisher=True, nan_policy="omit") > max_kurtosis:
        return False
    return True

def _is_bimodal_gmm(z: np.ndarray, bic_threshold: float = 10.0, min_mean_diff: float = 0.5, max_weight_ratio: float = 10.0) -> bool:
    """Enhanced GMM bimodality test using BIC + component separation + balanced weight."""
    z = z.reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, random_state=0).fit(z)
    gmm2 = GaussianMixture(n_components=2, random_state=0).fit(z)
    delta_bic = gmm1.bic(z) - gmm2.bic(z)

    if delta_bic < bic_threshold:
        return False

    means = gmm2.means_.flatten()
    weights = gmm2.weights_
    mean_diff = np.abs(means[0] - means[1])

    # condition 1: the difference between the two components should be greater than min_mean_diff
    if mean_diff < min_mean_diff:
        return False

    # condition 2: the weight of the two components should not be too unbalanced
    weight_ratio = max(weights) / min(weights)
    if weight_ratio > max_weight_ratio:
        return False

    return True

def _intersection(m1, s1, w1, m2, s2, w2) -> float:
    """Return the density-intersection of two 1-D Gaussians."""
    # Solve:  w1·N(m1,s1^2) == w2·N(m2,s2^2)
    a = 1.0 / (2 * s1**2) - 1.0 / (2 * s2**2)
    b = m2 / (s2**2) - m1 / (s1**2)
    c = (m1**2) / (2 * s1**2) - (m2**2) / (2 * s2**2) - np.log((s2 * w1) / (s1 * w2))
    roots = np.roots([a, b, c])
    # choose the root between means (if any)
    valid = roots[(roots > min(m1, m2)) & (roots < max(m1, m2))]
    return valid[0].real if valid.size else np.nan


def compute_pseudo_nonzero_gmm(
    adata: AnnData,
    *,
    layer: Optional[str] = None,
    max_iter: int = 100,
    random_state: int = 0,
    fallback_tau: float = -2.0,
    return_tau: bool = False,
) -> pd.Series | tuple[pd.Series, pd.Series]:
    """
    Adaptive pseudo non-zero ratio using a 2-component Gaussian mixture.

    Each gene is modelled as a mixture of *background* and *expressed* Gaussians.
    τ_g is the density intersection of the two components.

    Parameters
    ----------
    adata : AnnData
        Z-scored expression matrix.
    layer : str, optional
        Layer to use instead of ``adata.X``.
    max_iter, random_state
        Passed to :class:`sklearn.mixture.GaussianMixture`.
    fallback_tau : float
        τ_g to use if GMM becomes ill-conditioned or unimodal.
    return_tau : bool
        If ``True``, also return a Series of τ_g for inspection.

    Returns
    -------
    pseudo : pandas.Series
        Fraction of cells with expression > τ_g (index = genes).
    (pseudo, tau) : tuple
        Returned when ``return_tau=True``.
    """
    X = (adata.layers[layer] if layer else adata.X).toarray()
    n_cells, n_genes = X.shape

    ratios = np.empty(n_genes)
    taus   = np.empty(n_genes)

    for j in range(n_genes):
        z = X[:, j].reshape(-1, 1)
        try:
            gmm = GaussianMixture(
                n_components=2,
                max_iter=max_iter,
                covariance_type="full",
                random_state=random_state,
            ).fit(z)
            m = gmm.means_.flatten()
            s = np.sqrt(gmm.covariances_.flatten())
            w = gmm.weights_
            # sort by mean
            idx = np.argsort(m)
            tau = _intersection(m[idx[0]], s[idx[0]], w[idx[0]],
                                m[idx[1]], s[idx[1]], w[idx[1]])
            if np.isnan(tau):
                raise ValueError("intersection failed")
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"GMM failed for gene {adata.var_names[j]}: {e}")
            tau = fallback_tau

        taus[j]   = tau
        ratios[j] = (z[:, 0] > tau).mean()

    pseudo = pd.Series(ratios, index=adata.var_names, name="pseudo_nonzero_gmm")
    tau_s  = pd.Series(taus,   index=adata.var_names, name="tau_gmm")

    return (pseudo, tau_s) if return_tau else pseudo

def compute_pseudo_nonzero_otsu(
    adata: AnnData,
    *,
    layer: Optional[str] = None,
    nbins: int = 256,
    return_tau: bool = False,   
) -> pd.Series | Tuple[pd.Series, pd.Series]:
    """
    Adaptive pseudo non-zero ratio using Otsu's threshold.

    For each gene, τ_g is chosen to minimise intra-class variance between
    (z ≤ τ_g) and (z > τ_g).

    Parameters
    ----------
    adata : AnnData
        Z-scored expression matrix.
    layer : str, optional
        Layer to use instead of ``adata.X``.
    nbins : int
        Histogram bins used by :func:`skimage.filters.threshold_otsu`.

    Returns
    -------
    pseudo : pandas.Series
        Fraction of cells with expression > τ_g (index = genes).
    (pseudo, tau) : tuple
        Returned when ``return_tau=True``.
    """
    X = (adata.layers[layer] if layer else adata.X).toarray()
    ratios = [
        (x > threshold_otsu(x, nbins=nbins)).mean() if np.var(x) else 0.0
        for x in X.T
    ]
    taus = [threshold_otsu(x, nbins=nbins) if np.var(x) else 0.0 for x in X.T]
    tau_s = pd.Series(taus, index=adata.var_names, name="tau_otsu")
    return (pd.Series(ratios, index=adata.var_names, name="pseudo_nonzero_otsu"), tau_s) if return_tau else pd.Series(ratios, index=adata.var_names, name="pseudo_nonzero_otsu")

def compute_pseudo_nonzero_hybrid(
    adata: AnnData,
    *,
    layer: Optional[str] = None,
    fallback_tau: float = -2.0,
    bic_threshold: float = 10.0,
    return_tau: bool = False,
) -> pd.Series | Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Hybrid strategy for computing pseudo non-zero ratios per gene.

    - Try GMM if gene appears bimodal (ΔBIC > threshold).
    - Fall back to Otsu if unimodal.
    - Use fixed τ if variance too low or Otsu fails.

    Parameters
    ----------
    adata : AnnData
        AnnData object with z-scored expression matrix.
    layer : str, optional
        Use this layer instead of `adata.X`.
    fallback_tau : float
        Default τ if both methods fail.
    bic_threshold : float
        Minimum ΔBIC to consider GMM meaningful.
    return_tau : bool
        Return the τ used for each gene (optional).

    Returns
    -------
    pseudo : Series
        Pseudo non-zero ratio per gene.
    tau : Series (optional)
        Used τ per gene.
    method : Series (optional)
        Method used: 'gmm', 'otsu', or 'fallback'.
    """
    X = (adata.layers[layer] if layer else adata.X).toarray()
    n_genes = X.shape[1]

    pseudo_vals = np.zeros(n_genes)
    tau_vals = np.zeros(n_genes)
    method_vals = []

    for j in range(n_genes):
        z = X[:, j].reshape(-1, 1)
        var = np.var(z)
        gene = adata.var_names[j]

        # --- case 0: almost constant expression ---
        if var < 1e-4:
            tau, method = fallback_tau, "fallback"

        # --- case 1: approximately unimodal symmetric normal, directly fallback ---
        elif _is_uni_normal(z):
            tau, method = fallback_tau, "uni-normal"

        # --- case 2: try GMM bimodal ---
        elif _is_bimodal_gmm(z, bic_threshold):
            try:
                gmm = GaussianMixture(n_components=2, random_state=0).fit(z)
                m = gmm.means_.flatten(); s = np.sqrt(gmm.covariances_.flatten()); w = gmm.weights_
                idx = np.argsort(m)
                tau = _intersection(m[idx[0]], s[idx[0]], w[idx[0]],
                                    m[idx[1]], s[idx[1]], w[idx[1]])
                method = "gmm" if not np.isnan(tau) else "fallback"
            except Exception:
                tau, method = fallback_tau, "fallback"

        # --- case 3: use Otsu, but add τ upper limit ---
        else:
            try:
                tau = threshold_otsu(z[:, 0])
                # protection: if τ > 2.5 (or >1.5×σ), considered meaningless → fallback
                if tau > 2.5:
                    raise ValueError("Otsu τ too high")
                method = "otsu"
            except Exception:
                tau, method = fallback_tau, "fallback"

        ratio = (z[:, 0] > tau).mean()
        pseudo_vals[j] = ratio
        tau_vals[j] = tau
        method_vals.append(method)

    pseudo = pd.Series(pseudo_vals, index=adata.var_names, name="pseudo_nonzero_hybrid")
    tau_series = pd.Series(tau_vals, index=adata.var_names, name="tau_hybrid")
    method_series = pd.Series(method_vals, index=adata.var_names, name="method")

    return (pseudo, tau_series, method_series) if return_tau else pseudo

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

