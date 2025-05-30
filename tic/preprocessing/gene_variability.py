import warnings
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData
from scipy.sparse import issparse


def filter_cells_by_type(
    adata: AnnData,
    cell_types: Union[str, List[str]]
) -> AnnData:
    """
    Subset an AnnData object by one or multiple cell types.

    Parameters
    ----------
    adata
        AnnData with `.obs['cell_type']` annotation.
    cell_types
        A single cell type (str) or list of cell types to keep.

    Returns
    -------
    AnnData
        A new AnnData containing only the specified cell types.
    """
    if isinstance(cell_types, str):
        mask = adata.obs['cell_type'] == cell_types
    else:
        mask = adata.obs['cell_type'].isin(cell_types)
    return adata[mask].copy()


def compute_gene_variability(
    adata: AnnData,
    gene_list: List[str]
) -> pd.Series:
    """
    Compute variance of expression for a given list of genes,
    ignoring any genes not present in the AnnData.

    Parameters
    ----------
    adata
        AnnData whose `.var_names` include at least one gene from `gene_list`.
    gene_list
        List of gene names to compute variability for.

    Returns
    -------
    pd.Series
        Variance of each valid gene across all cells, indexed by gene name.

    Raises
    ------
    ValueError
        If none of the genes in `gene_list` are found in `adata.var_names`.
    """
    present = [g for g in gene_list if g in adata.var_names]
    missing = set(gene_list) - set(present)

    if not present:
        raise ValueError(
            "None of the requested genes are in the AnnData object."
        )

    if missing:
        warnings.warn(
            f"Ignoring {len(missing)} unknown gene(s): {sorted(missing)}",
            UserWarning
        )

    adata_sub = adata[:, present]
    X = adata_sub.X
    if issparse(X):
        X = X.toarray()
    variances = np.var(X, axis=0)
    return pd.Series(data=variances, index=present, name='variance')


def plot_distribution_rug(
    all_vars: pd.Series,
    sel_vars: pd.Series,
    title: Optional[str] = None,
    bins: int = 100,
    figsize: tuple = (8, 4)
) -> None:
    """
    Plot a histogram of all gene variances with a "rug" of selected genes.

    Parameters
    ----------
    all_vars
        Variance series for all genes.
    sel_vars
        Variance series for selected genes.
    title
        Optional title for the plot.
    bins
        Number of bins for the histogram.
    figsize
        Figure size tuple (width, height).
    """
    plt.figure(figsize=figsize)
    counts, edges, _ = plt.hist(
        all_vars, bins=bins, alpha=0.4, label='All genes'
    )
    ylim = plt.ylim()
    for gene, var in sel_vars.items():
        plt.vlines(var, ymin=0, ymax=ylim[1] * 0.05, color='C1')
        plt.text(
            var, ylim[1] * 0.055, gene,
            rotation=90, va='bottom', fontsize=8
        )
    plt.xlabel('Variance')
    plt.ylabel('Count')
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_bar_with_rank(
    all_vars: pd.Series,
    sel_vars: pd.Series,
    title: Optional[str] = None,
    figsize: tuple = (8, 4)
) -> None:
    """
    Bar plot of selected genes with their global variance rank.

    Parameters
    ----------
    all_vars
        Variance series for all genes.
    sel_vars
        Variance series for selected genes.
    title
        Optional title for the plot.
    figsize
        Figure size tuple (width, height).
    """
    ranks = all_vars.rank(ascending=False, method='min').astype(int)
    sel_ranks = ranks.loc[sel_vars.index]
    sel_sorted = sel_vars.sort_values(ascending=False)

    plt.figure(figsize=figsize)
    ax = sel_sorted.plot.bar()
    for i, gene in enumerate(sel_sorted.index):
        ax.text(
            i, sel_sorted[gene] * 1.02,
            f"#{sel_ranks[gene]}", ha='center', va='bottom',
            fontsize=9
        )
    plt.ylabel('Variance')
    plt.xlabel('Gene')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_scatter_rank(
    all_vars: pd.Series,
    sel_vars: pd.Series,
    title: Optional[str] = None,
    figsize: tuple = (8, 4)
) -> None:
    """
    Scatter plot of variance vs. rank for all genes, highlighting selected ones.

    Parameters
    ----------
    all_vars
        Variance series for all genes.
    sel_vars
        Variance series for selected genes.
    title
        Optional title for the plot.
    figsize
        Figure size tuple (width, height).
    """
    sorted_all = all_vars.sort_values(ascending=False)
    x_all = np.arange(len(sorted_all))
    ranks = sorted_all.rank(ascending=False, method='min').astype(int)

    plt.figure(figsize=figsize)
    plt.scatter(x_all, sorted_all, s=5, color='lightgray', label='All genes')

    for gene, var in sel_vars.items():
        r = ranks[gene] - 1
        plt.scatter(r, var, s=40, label=gene)
        plt.text(r, var, gene, fontsize=8, ha='right', va='bottom')

    plt.xlabel('Rank')
    plt.ylabel('Variance')
    if title:
        plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_violin_points(
    all_vars: pd.Series,
    sel_vars: pd.Series,
    title: Optional[str] = None,
    figsize: tuple = (6, 4)
) -> None:
    """
    Violin plot of all gene variances with overlaid points for selected genes.

    Parameters
    ----------
    all_vars
        Variance series for all genes.
    sel_vars
        Variance series for selected genes.
    title
        Optional title for the plot.
    figsize
        Figure size tuple (width, height).
    """
    plt.figure(figsize=figsize)
    parts = plt.violinplot(
        all_vars.values, showmeans=True, showextrema=True
    )
    for pc in parts['bodies']:
        pc.set_alpha(0.4)

    # Jitter selected points
    jitter = 0.04
    x_jitter = np.random.normal(0, jitter, size=len(sel_vars))
    plt.scatter(
        x_jitter, sel_vars.values,
        s=40, color='C1', edgecolor='k', alpha=0.8
    )
    for x, gene, var in zip(x_jitter, sel_vars.index, sel_vars.values):
        plt.text(x, var, gene, fontsize=8, ha='right', va='bottom')

    plt.xticks([])
    plt.ylabel('Variance')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def analyze_gene_variability(
    adata: AnnData,
    gene_list: List[str],
    cell_types: Optional[Union[str, List[str]]] = None,
    plot: bool = True,
    viz_styles: Union[str, List[str]] | Literal['histogram_rug', 'bar_rank', 'scatter_rank', 'violin_points'] = 'bar_rank'
) -> pd.Series:
    """
    Full pipeline: filter by cell type, compute gene variances,
    and visualize using one or more selected styles.

    Available viz_styles:
      - "histogram_rug"
      - "bar_rank"
      - "scatter_rank"
      - "violin_points"

    Parameters
    ----------
    adata
        AnnData with expression matrix `.X` and `.obs['cell_type']`.
    gene_list
        List of gene names to analyze.
    cell_types
        Cell type or list of cell types to filter by. If None,
        all cells are used.
    plot
        Whether to plot the results.
    viz_styles
        A style or list of styles for visualization.(works when plot is True)

    Returns
    -------
    pd.Series
        Gene variances indexed by gene name.
    """
    # Step 1: filter by cell type
    if cell_types is not None:
        adata = filter_cells_by_type(adata, cell_types)

    # Step 2: compute variances for selected genes
    sel_vars = compute_gene_variability(adata, gene_list)

    # Prepare viz_styles list
    if isinstance(viz_styles, str):
        viz_styles = [viz_styles]
    valid = {
        'histogram_rug',
        'bar_rank',
        'scatter_rank',
        'violin_points'
    }
    for style in viz_styles:
        if style not in valid:
            raise ValueError(f"Unknown viz style: {style}")

    # Compute all gene variances once if needed
    if any(s in viz_styles for s in valid - {'bar_rank'}):
        all_vars = compute_gene_variability(adata, list(adata.var_names))

    title_base = "Gene variance"
    if cell_types is not None:
        title_base += f" in {cell_types}"

    # Step 3: visualize
    if plot:
        for style in viz_styles:
            title = f"{title_base}: {style.replace('_', ' ').title()}"
            if style == 'histogram_rug':
                plot_distribution_rug(all_vars, sel_vars, title=title)
            elif style == 'bar_rank':
                plot_bar_with_rank(all_vars, sel_vars, title=title)
            elif style == 'scatter_rank':
                plot_scatter_rank(all_vars, sel_vars, title=title)
            elif style == 'violin_points':
                plot_violin_points(all_vars, sel_vars, title=title)

    return sel_vars