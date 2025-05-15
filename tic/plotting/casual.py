"""
Plotting utilities for causal inference results stored in an AnnData object.

Every function expects `adata.uns["causal_results"]` to be a dict mapping
predictor names (formatted as "CellType&Gene") to result dicts produced by
CausalWrapper, where each result dict contains at least:
  - "p_value": raw p-value
  - "best_model_parameters": list of coefficients
  - "best_lag": integer lag value
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


def bonferroni(p_raw: float | np.ndarray, n_tests: int) -> float | np.ndarray:
    """
    Apply Bonferroni correction to p-value(s) and cap at 1.0.

    Parameters
    ----------
    p_raw : float or np.ndarray
        Raw p-value or array of raw p-values.
    n_tests : int
        Number of independent tests.

    Returns
    -------
    float or np.ndarray
        Bonferroni-adjusted p-value(s).
    """
    return np.minimum(np.asarray(p_raw) * n_tests, 1.0)


def _extract_matrix(
    results: Dict[str, Dict],
    metric: str,
    n_tests: int,
    log2_transform: bool,
    highlight_pval: float,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], bool]]:
    """
    Build a pivot table [cell_type x biomarker] of the chosen metric.

    Applies Bonferroni correction to each p-value and filters by threshold.

    Parameters
    ----------
    results
        Mapping "CellType&Gene" → result dict.
    metric
        One of "max_abs", "rsm", or "p_value".
    n_tests
        Number of predictors to use for Bonferroni correction.
    log2_transform
        Whether to log2-transform effect-size metrics.
    highlight_pval
        p-value below which to mark significance.

    Returns
    -------
    matrix
        DataFrame indexed by cell type, columns = biomarkers, values = metric.
    sig_map
        Dict mapping (cell_type, biomarker) to True if p < highlight_pval.
    """
    data: List[Tuple[str, str, float]] = []
    sig_map: Dict[Tuple[str, str], bool] = {}

    for key, res in results.items():
        if "&" not in key:
            continue  # skip malformed keys

        cell, gene = key.split("&", 1)

        # Bonferroni-corrected p-value
        raw_p = res.get("p_value", 1.0)
        corr_p = bonferroni(raw_p, n_tests)
        if corr_p >= 1.0: # if adjusted p-value is greater than 1, skip
            continue

        if metric == "p_value":
            value = -np.log10(corr_p + 1e-12)
            sig_map[(cell, gene)] = (corr_p < highlight_pval)
        else:
            coeffs = res.get("best_model_parameters", [])
            lag = res.get("best_lag", 0)
            # coefficients at the selected lag
            segment = coeffs[lag : lag * 2] if lag and len(coeffs) >= lag * 2 else []
            if not segment:
                continue

            if metric == "max_abs":
                value = max(abs(x) for x in segment)
            elif metric == "rsm":
                value = np.sqrt(np.mean(np.square(segment)))
            else:
                raise ValueError("metric must be 'max_abs', 'rsm', or 'p_value'")

            if log2_transform:
                value = np.log2(value + 1e-12)

        data.append((cell, gene, float(value)))

    if not data:
        return pd.DataFrame(), {}

    df = pd.DataFrame(data, columns=["celltype", "biomarker", "value"])
    matrix = df.pivot(index="celltype", columns="biomarker", values="value")
    return matrix, sig_map


def plot_causal_heatmap(
    adata: anndata.AnnData,
    metric: str = "max_abs",
    log2_transform: bool = True,
    fillna_val: float = 0.0,
    highlight_pval: float = 0.05,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Causal Heatmap",
    celltype_grouping: Optional[Dict[str, List[str]]] = None,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Draw a hierarchical or grouped heatmap of causal metrics.

    Parameters
    ----------
    adata
        AnnData with `adata.uns["causal_results"]` present.
    metric
        Metric to plot: "max_abs", "rsm", or "p_value".
    log2_transform
        Apply log2 to effect-size metrics.
    fillna_val
        Value to fill for missing entries in the matrix.
    highlight_pval
        Mark entries with p < highlight_pval with an asterisk.
    figsize
        Figure size.
    celltype_grouping
        Optional mapping of group names → list of cell types for custom ordering.
    save_path
        If provided, path to save the figure (before show).
    """
    if "causal_results" not in adata.uns:
        raise ValueError("adata.uns lacks 'causal_results'.")

    results = adata.uns["causal_results"]
    if not isinstance(results, dict):
        raise TypeError("`causal_results` must be a dict.")

    # Number of tests = total predictors
    n_tests = len(results)

    matrix, sig_map = _extract_matrix(
        results, metric, n_tests, log2_transform, highlight_pval
    )
    if all(not is_sig for is_sig in sig_map.values()):
        print("No statistically significant predictors found (p ≥ {:.3f}).".format(highlight_pval))
        return

    if matrix.empty:
        print("No valid predictors after filtering; nothing to plot.")
        return

    matrix = matrix.fillna(fillna_val)
    label_map = {
        "max_abs": "Max |Coeff|",
        "rsm": "RSM(Coeff)",
        "p_value": "-log10(adj p)",
    }
    metric_label = label_map.get(metric, metric)
    if metric != "p_value" and log2_transform:
        metric_label = f"log2({metric_label})"

    plt.figure(figsize=figsize)

    if celltype_grouping:
        # reorder rows according to groups
        ordered = [ct for group in celltype_grouping.values() for ct in group]
        matrix = matrix.reindex(ordered)

        ax = sns.heatmap(
            matrix,
            cmap="coolwarm",
            linewidths=0.5,
            cbar_kws={"label": metric_label},
            xticklabels=matrix.shape[1] <= 20,
            yticklabels=matrix.shape[0] <= 20,
        )
        ax.set_title(title, fontsize=14)
    else:
        cg = sns.clustermap(
            matrix,
            cmap="coolwarm",
            linewidths=0.5,
            figsize=figsize,
            cbar_kws={"label": metric_label},
            xticklabels=matrix.shape[1] <= 20,
            yticklabels=matrix.shape[0] <= 20,
        )
        ax = cg.ax_heatmap
        ax.set_title(title, pad=80)

    # annotate significance for p-value metric
    if metric == "p_value":
        for (cell, gene), is_sig in sig_map.items():
            if not is_sig or cell not in matrix.index or gene not in matrix.columns:
                continue
            y = list(matrix.index).index(cell)
            x = list(matrix.columns).index(gene)
            ax.text(x + 0.5, y + 0.5, "*", ha="center", va="center", color="black")

    plt.tight_layout()
    if save_path:
        # make the directory if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    return plt.gca()


def plot_causal_bar(
    adata: anndata.AnnData,
    top_n: int = 10,
    pval_threshold: float = 0.05,
    title: str = "Causal Bar Plot",
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plot the top predictors by Bonferroni-adjusted p-value as a horizontal bar chart.

    Parameters
    ----------
    adata
        AnnData with `adata.uns["causal_results"]` present.
    top_n
        Number of top predictors to display.
    pval_threshold
        Significance threshold to draw a vertical line.
    save_path
        If provided, path to save the figure (before show).
    """
    if "causal_results" not in adata.uns:
        raise ValueError("adata.uns lacks 'causal_results'.")

    results = adata.uns["causal_results"]
    if not isinstance(results, dict):
        raise TypeError("`causal_results` must be a dict.")

    predictor_names = list(results.keys())
    n_tests = len(predictor_names)
    raw_ps = [results[name].get("p_value", 1.0) for name in predictor_names]
    adj_ps = bonferroni(raw_ps, n_tests)

    # sort and select top_n
    idx = np.argsort(adj_ps)[:top_n]
    top_preds = [predictor_names[i] for i in idx]
    top_ps = adj_ps[idx]

    plt.figure(figsize=(8, max(2, top_n * 0.5)))
    plt.barh(range(len(top_preds)), top_ps, color="skyblue")
    plt.yticks(range(len(top_preds)), top_preds)
    plt.xlabel("Bonferroni-adjusted p-value")
    plt.title(title)
    plt.axvline(pval_threshold, color="red", ls="--", label=f"p = {pval_threshold}")
    plt.legend()
    plt.tight_layout()

    if save_path:
        # make the directory if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    return plt.gca()


def plot_causal_volcano(
    adata: anndata.AnnData,
    x_metric: str = "max_abs",
    log2_thresh: float = 1.0,
    pval_threshold: float = 0.05,
    top_n_label: int = 3,
    title: str = "Causal Volcano Plot",
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Create a volcano plot of causal inference: log2(effect size) vs. -log10(adj p-value).

    Parameters
    ----------
    adata
        AnnData with `adata.uns["causal_results"]` present.
    x_metric
        Effect-size metric: "max_abs" or "rsm".
    log2_thresh
        Threshold on log2(effect size) for coloring.
    pval_threshold
        Significance threshold for the horizontal line.
    top_n_label
        Number of most significant points to annotate.
    save_path
        If provided, path to save the figure (before show).
    """
    if "causal_results" not in adata.uns:
        raise ValueError("adata.uns lacks 'causal_results'.")

    results = adata.uns["causal_results"]
    if not isinstance(results, dict):
        raise TypeError("`causal_results` must be a dict.")

    names = list(results.keys())
    n_tests = len(names)

    xs, ys, ps, labels = [], [], [], []
    for name in names:
        res = results[name]
        raw_p = res.get("p_value", 1.0)
        adj_p = bonferroni(raw_p, n_tests)
        if adj_p >= 1.0:
            continue

        # y-axis: -log10(p)
        y = -np.log10(adj_p + 1e-12)

        # compute effect size
        coeffs = res.get("best_model_parameters", [])
        lag = res.get("best_lag", 0)
        segment = coeffs[lag : lag * 2] if lag and len(coeffs) >= lag * 2 else []
        if not segment:
            continue

        if x_metric == "max_abs":
            effect = max(abs(v) for v in segment)
        elif x_metric == "rsm":
            effect = np.sqrt(np.mean(np.square(segment)))
        else:
            raise ValueError("x_metric must be 'max_abs' or 'rsm'")

        x = np.log2(effect + 1e-12)

        xs.append(x)
        ys.append(y)
        ps.append(adj_p)
        labels.append(name)

    if not xs:
        print("No valid causal results to plot.")
        return

    xs = np.array(xs)
    ys = np.array(ys)
    ps = np.array(ps)

    # scatter
    cmap = cm.get_cmap("coolwarm_r")
    norm = Normalize(vmin=0.0, vmax=pval_threshold)

    plt.figure(figsize=(8, 6))
    for x, y, p in zip(xs, ys, ps):
        color = cmap(norm(p)) if abs(x) > log2_thresh else "lightgray"
        plt.scatter(x, y, color=color, s=30, alpha=0.8)

    # annotate top points by p-value
    top_idx = np.argsort(ys)[-top_n_label:]
    for i in top_idx:
        plt.annotate(
            labels[i],
            (xs[i], ys[i]),
            textcoords="offset points",
            xytext=(-5, 5),
            ha="right",
            fontsize=8,
        )

    # threshold lines
    plt.axhline(-np.log10(pval_threshold), color="red", ls="--", lw=1)
    plt.axvline(-log2_thresh, color="gray", ls="--", lw=1)
    plt.axvline(log2_thresh, color="gray", ls="--", lw=1)

    plt.xlabel("log2(effect size)")
    plt.ylabel("-log10(Bonferroni-adjusted p)")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        # make the directory if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    return plt.gca()