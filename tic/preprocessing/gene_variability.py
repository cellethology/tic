# tic/preprocessing/gene_variability.py
"""
Gene variability analysis utilities for the TIC pipeline.

This module provides a full pipeline to:
1.  Optionally subset an :class:`~anndata.AnnData` object by cell type(s).
2.  Compute **per-gene expression variance** across cells.
3.  Classify user-specified genes as *highly variable* or not, based on either
    a global *top-N* rank or a variance *quantile* threshold.
4.  Visualise the variance distribution with multiple styles to help assess
    whether the genes of interest (GOIs) truly vary in the current dataset – a
    prerequisite for meaningful pseudotime ordering.

Why this matters
----------------
If a GOI exhibits little variation, any ordering of cells (e.g. along
pseudotime) will look *flat* for that gene, rendering biological conclusions
weak.  Detecting low variability early lets you decide whether to drop the gene
or adjust experimental design.
"""
from __future__ import annotations

import warnings
from typing import Literal, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData
from scipy.sparse import issparse

# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------

def filter_cells_by_type(
    adata: AnnData,
    cell_types: Union[str, Sequence[str]],
) -> AnnData:
    """Return a slice of *adata* containing only *cell_types*.

    Parameters
    ----------
    adata
        Input AnnData with an ``.obs['cell_type']`` column.
    cell_types
        Single cell type or iterable of cell types to *keep*.
    """
    mask = (
        adata.obs["cell_type"].isin([cell_types])
        if isinstance(cell_types, str)
        else adata.obs["cell_type"].isin(cell_types)
    )
    return adata[mask].copy()


def _extract_dense(X):
    """Return a dense *numpy* view of ``X`` (handles sparse inputs)."""
    return X.toarray() if issparse(X) else X


def compute_gene_variability(
    adata: AnnData,
    gene_list: Sequence[str],
) -> pd.Series:
    """Compute *per-gene* variance for *gene_list* (silently skips missing).

    Notes
    -----
    Returned series is **not** log-transformed.  Make sure *adata.X* contains
    the expression scale you care about (raw counts, log1p, etc.).
    """
    present = [g for g in gene_list if g in adata.var_names]
    missing = set(gene_list) - set(present)
    if not present:
        raise ValueError("None of the requested genes are present in adata.")
    if missing:
        warnings.warn(
            f"Ignoring {len(missing)} unknown gene(s): {sorted(missing)}",
            UserWarning,
        )

    X = _extract_dense(adata[:, present].X)
    variances = np.var(X, axis=0, ddof=0)
    return pd.Series(variances, index=present, name="variance")


# -----------------------------------------------------------------------------
# HVG classification
# -----------------------------------------------------------------------------

def classify_gene_variability(
    all_vars: pd.Series,
    sel_vars: pd.Series,
    *,
    top_n: int | None = 2000,
    quantile: float | None = None,
) -> pd.DataFrame:
    """Determine whether each selected gene is *highly variable*.

    Exactly **one** of *top_n* **or** *quantile* may be provided.

    Parameters
    ----------
    all_vars
        Variance of **all** genes (index must be gene names).
    sel_vars
        Variance of user-selected genes (subset of *all_vars.index*).
    top_n
        Genes with global variance rank ≤ *top_n* are marked as HVG.
    quantile
        Alternatively, genes with variance ≥ *all_vars.quantile(quantile)* are
        marked as HVG.

    Returns
    -------
    pd.DataFrame
        Columns: ``variance``, ``rank``, ``is_hvg`` (bool).
    """
    if (top_n is None) == (quantile is None):
        raise ValueError("Specify exactly one of 'top_n' or 'quantile'.")

    ranks = all_vars.rank(ascending=False, method="min").astype(int)
    sel_ranks = ranks.loc[sel_vars.index]

    if top_n is not None:
        threshold_flag = sel_ranks <= top_n
    else:
        var_threshold = all_vars.quantile(quantile)
        threshold_flag = sel_vars >= var_threshold

    return pd.DataFrame(
        {
            "variance": sel_vars,
            "rank": sel_ranks,
            "is_hvg": threshold_flag,
        }
    )


# -----------------------------------------------------------------------------
# Visualisation helpers (unchanged except small refactor)
# -----------------------------------------------------------------------------

def _setup_ax(figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_distribution_rug(
    all_vars: pd.Series,
    sel_vars: pd.Series,
    *,
    title: str | None = None,
    bins: int = 100,
    figsize: tuple[int, int] = (8, 4),
):
    fig, ax = _setup_ax(figsize)
    ax.hist(all_vars, bins=bins, alpha=0.4, label="All genes")
    ylim = ax.get_ylim()
    for gene, var in sel_vars.items():
        ax.vlines(var, ymin=0, ymax=ylim[1] * 0.05, color="C1")
        ax.text(var, ylim[1] * 0.055, gene, rotation=90, va="bottom", fontsize=8)
    ax.set_xlabel("Variance")
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()


def plot_bar_with_rank(
    all_vars: pd.Series,
    sel_vars: pd.Series,
    *,
    title: str | None = None,
    figsize: tuple[int, int] = (8, 4),
):
    ranks = all_vars.rank(ascending=False, method="min").astype(int)
    sel_ranks = ranks.loc[sel_vars.index]
    sel_sorted = sel_vars.sort_values(ascending=False)

    fig, ax = _setup_ax(figsize)
    sel_sorted.plot.bar(ax=ax)
    for i, gene in enumerate(sel_sorted.index):
        ax.text(i, sel_sorted[gene] * 1.02, f"#{sel_ranks[gene]}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Variance")
    ax.set_xlabel("Gene")
    if title:
        ax.set_title(title)
    fig.tight_layout()


def plot_scatter_rank(
    all_vars: pd.Series,
    sel_vars: pd.Series,
    *,
    title: str | None = None,
    figsize: tuple[int, int] = (8, 4),
):
    sorted_all = all_vars.sort_values(ascending=False)
    x_all = np.arange(len(sorted_all))
    ranks = sorted_all.rank(ascending=False, method="min").astype(int)

    fig, ax = _setup_ax(figsize)
    ax.scatter(x_all, sorted_all, s=5, color="lightgray", label="All genes")

    for gene, var in sel_vars.items():
        r = ranks[gene] - 1
        ax.scatter(r, var, s=40, label=gene)
        ax.text(r, var, gene, fontsize=8, ha="right", va="bottom")

    ax.set_xlabel("Rank")
    ax.set_ylabel("Variance")
    if title:
        ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()


def plot_violin_points(
    all_vars: pd.Series,
    sel_vars: pd.Series,
    *,
    title: str | None = None,
    figsize: tuple[int, int] = (6, 4),
):
    fig, ax = _setup_ax(figsize)
    parts = ax.violinplot(all_vars.values, showmeans=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.4)

    jitter = np.random.normal(0, 0.04, size=len(sel_vars))
    ax.scatter(jitter, sel_vars.values, s=40, color="C1", edgecolor="k", alpha=0.8)
    for x, gene, var in zip(jitter, sel_vars.index, sel_vars.values):
        ax.text(x, var, gene, fontsize=8, ha="right", va="bottom")

    ax.set_xticks([])
    ax.set_ylabel("Variance")
    if title:
        ax.set_title(title)
    fig.tight_layout()


# -----------------------------------------------------------------------------
# High-level API
# -----------------------------------------------------------------------------

def analyze_gene_variability(
    adata: AnnData,
    gene_list: Sequence[str],
    *,
    cell_types: Union[str, Sequence[str], None] = None,
    classify_top_n: int | None = 2000,
    classify_quantile: float | None = None,
    plot: bool = True,
    viz_styles: Union[
        str,
        Sequence[str],
        Literal["histogram_rug", "bar_rank", "scatter_rank", "violin_points"],
    ] = "bar_rank",
) -> pd.DataFrame:
    """Full pipeline for gene variability *assessment* and *visualisation*.

    Parameters
    ----------
    adata
        Input AnnData.  **Ensure** that ``adata.X`` is already normalised /
        transformed as required (e.g. log1p of total-normalised counts).
    gene_list
        Genes of interest (GOIs) whose variability you want to test.
    cell_types
        If provided, restrict analysis to these cell types.
    classify_top_n, classify_quantile
        Criteria for labelling a gene as *highly variable*.  Specify **one** of
        them (defaults to ``top_n=2000``).
    plot
        Whether to render variance distribution plots.
    viz_styles
        One or more plot styles (ignored if ``plot=False``).

    Returns
    -------
    pd.DataFrame
        ``['variance', 'rank', 'is_hvg']`` for each GOI.
    """
    # 1) Subset by cell type (optional)
    if cell_types is not None:
        adata = filter_cells_by_type(adata, cell_types)

    # 2) Compute variance for *all* genes and for GOIs
    all_vars = compute_gene_variability(adata, list(adata.var_names))
    sel_vars = compute_gene_variability(adata, gene_list)

    # 3) Classify HVGs
    hvg_df = classify_gene_variability(
        all_vars,
        sel_vars,
        top_n=classify_top_n,
        quantile=classify_quantile,
    )

    # 4) Plotting
    if plot:
        if isinstance(viz_styles, str):
            viz_styles = [viz_styles]
        valid_styles = {
            "histogram_rug",
            "bar_rank",
            "scatter_rank",
            "violin_points",
        }
        for s in viz_styles:
            if s not in valid_styles:
                raise ValueError(f"Unknown viz style: {s}")

        title_base = "Gene variance"
        if cell_types is not None:
            title_base += f" in {cell_types}"

        for s in viz_styles:
            title = f"{title_base}: {s.replace('_', ' ').title()}"
            if s == "histogram_rug":
                plot_distribution_rug(all_vars, sel_vars, title=title)
            elif s == "bar_rank":
                plot_bar_with_rank(all_vars, sel_vars, title=title)
            elif s == "scatter_rank":
                plot_scatter_rank(all_vars, sel_vars, title=title)
            elif s == "violin_points":
                plot_violin_points(all_vars, sel_vars, title=title)

    return hvg_df
