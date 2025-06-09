# ==============================================================
# File: tic/plotting/variability.py
# --------------------------------------------------------------
"""Plotting helper: variability vs. gene rank with category markers in a Nature-like style.

Example usage
-------------
>>> from tic.data import load_xenium_dataset, load_region
>>> from tic.metrics import compute_variability
>>> from tic.plotting import plot_variability
>>> ad = load_xenium_dataset("xenium_ffpe_human_breast", normalize="scanpy", log=True)
>>> var = compute_variability(ad, dataset_type="xenium", metrics=["cv"])
>>> plot_variability({"breast": var["cv"]}, metric="cv", top_n=5)
"""
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Literal, Mapping, Optional, Sequence

__all__ = ["plot_variability"]

# Default color palette for datasets (can be replaced with a colorblind-friendly palette)
_DEFAULT_DATASET_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
]
# Default colors for gene categories
_DEFAULT_CATEGORY_COLORS = {
    "Epithelial": "#e377c2",   # pink
    "Mesenchymal": "#17becf",   # cyan
    "EMT Transcription Factors": "#8c564b",  # brown
}


def plot_variability(
    variability: Mapping[str, pd.Series] | Mapping[str, pd.DataFrame],
    *,
    metric: str | Literal["cv", "entropy", "kurtosis", "skewness", "mse", "gini"],
    gene_categories: Optional[Mapping[str, Sequence[str]]] = None,
    dataset_colors: Optional[Mapping[str, str]] = None,
    category_colors: Optional[Mapping[str, str]] = None,
    ax: Optional[plt.Axes] = None,
    y_max: Optional[float] = None,
    y_percentile_clip: Optional[float] = None,
    log_x: bool = False,
    log_y: bool = False,
    ascending: bool = False,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (7, 5),
    top_n: int = 0,
    annotate_top: bool = True,
    annotate_emt: bool = False,
) -> plt.Axes:
    """Plot variability metric vs. gene rank with category highlights and optional annotations.

    Parameters
    ----------
    variability : dict
        Mapping from dataset_name to Series/DataFrame of gene variability.
    metric : str
        Column to extract from DataFrame (if applicable).
    gene_categories : dict
        Category name → list of gene names.
    dataset_colors : dict
        Dataset name → color code.
    category_colors : dict
        Category name → color code.
    ax : matplotlib.axes.Axes
        Optional external Axes to draw on.
    y_max : float
        Maximum value for Y-axis.This is raw value before log transformation.
    y_percentile_clip : float
        Percentile to clip Y-axis.This is raw value before log transformation.
    log_x : bool
        Use log scale for X-axis.
    log_y : bool
        Use log scale for Y-axis.
    ascending : bool
        Whether to sort genes in ascending order.
        For CV, we want to sort genes in descending order.
        For kurtosis, skewness, we want to sort genes in ascending order: we want the gene more like a Uniform distribution 
            -> have values across the range instead of a sharp peak.
    title : str
        Optional title for the plot.
    figsize : tuple
        Figure size in inches.
    top_n : int
        Number of top genes to annotate.
    annotate_top : bool
        Whether to annotate top genes.
    annotate_emt : bool
        Whether to annotate genes from gene_categories.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STSong"],
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "axes.grid": False,
        "legend.frameon": False,
    })

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ds_colors: Dict[str, str] = {}
    if dataset_colors:
        ds_colors.update(dataset_colors)
    for i, ds in enumerate(variability.keys()):
        if ds not in ds_colors:
            ds_colors[ds] = _DEFAULT_DATASET_COLORS[i % len(_DEFAULT_DATASET_COLORS)]

    cat_colors = {**_DEFAULT_CATEGORY_COLORS, **(category_colors or {})}

    for idx, (ds_name, data) in enumerate(variability.items()):
        series = data[metric] if isinstance(data, pd.DataFrame) else data
        series_clean = series.dropna()
        if y_percentile_clip is not None:
            clip_value = np.percentile(series_clean.values, y_percentile_clip)
            series_clean = series_clean[series_clean <= clip_value]
        if y_max is not None:
            series_clean = series_clean[series_clean <= y_max]
        sorted_genes = series_clean.sort_values(ascending=ascending)
        ranks = pd.Series(np.arange(1, len(sorted_genes) + 1), index=sorted_genes.index)

        ax.plot(
            ranks.values,
            sorted_genes.values,
            label=ds_name,
            color=ds_colors[ds_name],
            linewidth=1.5,
            alpha=0.9,
        )

        if top_n > 0 and annotate_top:
            top_genes = list(sorted_genes.index[:top_n])
            top_ranks = ranks.loc[top_genes].values
            top_vals = sorted_genes.loc[top_genes].values
            for gene, x_pos, y_pos in zip(top_genes, top_ranks, top_vals):
                ax.text(
                    x_pos,
                    y_pos,
                    gene,
                    fontsize=8,
                    color=ds_colors[ds_name],
                    ha="left",
                    va="bottom",
                    rotation=30,
                    alpha=0.8,
                )

        if gene_categories:
            for cat, genes in gene_categories.items():
                common = list(set(genes).intersection(sorted_genes.index))
                if not common:
                    continue
                cat_ranks = ranks.loc[common]
                cat_values = sorted_genes.loc[common]
                ax.scatter(
                    cat_ranks.values,
                    cat_values.values,
                    color=cat_colors.get(cat, "black"),
                    marker="o",
                    s=60,
                    edgecolors="white",
                    linewidths=0.8,
                    label=f"{cat}" if idx == 0 else "_nolegend_",
                    zorder=5,
                )
                if annotate_emt:
                    for gene, x_pos, y_pos in zip(common, cat_ranks.values, cat_values.values):
                        ax.text(
                            x_pos,
                            y_pos,
                            gene,
                            fontsize=7,
                            color=cat_colors.get(cat, "black"),
                            ha="right",
                            va="top",
                            rotation=30,
                            alpha=0.7,
                        )
    
    if y_max is not None:
        ax.set_ylim(top=y_max)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    if metric == "kurtosis":
        # add a horizontal line at 3 or log(3) if log_y is True
        if log_y:
            ax.axhline(np.log(3), color="black", linestyle="--", linewidth=1, label="Kurtosis = 3")
        else:
            ax.axhline(3, color="black", linestyle="--", linewidth=1, label="Kurtosis = 3")

    ax.set_xlabel(f"Gene rank (lower is more variable)" if not log_x else "log(Gene rank)", fontsize=12)
    ax.set_ylabel(metric.upper() if not log_y else f"log({metric.upper()})", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")

    ax.legend(loc="upper right", fontsize=10, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    return ax