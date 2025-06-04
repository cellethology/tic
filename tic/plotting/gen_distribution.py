# ==============================================================
# File: tic/plotting/gen_distribution.py
# ==============================================================
"""
Plot gene expression distributions for multiple gene categories.

"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from typing import List, Mapping, Sequence, Literal, Optional
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from scipy.sparse import issparse

def plot_gene_distributions_by_category(
    adata: AnnData,
    gene_dict: Mapping[str, Sequence[str]],
    expression_key: Literal["X", "raw"] = "X",
    plot_type: Literal["violin", "box"] = "violin",
    max_genes_per_category: Optional[int] = 10,
    figsize_per_row: float = 1.8,
    title: Optional[str] = "Gene Expression Distributions",
):
    """
    Plot gene expression distributions for multiple gene categories.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing expression data.
    gene_dict : dict
        Mapping from category name to a list of gene names.
    expression_key : {"X", "raw"}, default="X"
        Whether to use `adata.X` or `adata.raw.X` for expression values.
    plot_type : {"violin", "box"}, default="violin"
        Whether to plot violin plots or boxplots.
    max_genes_per_category : int or None
        Maximum number of genes to plot per category.
    figsize_per_row : float
        Height per subplot row.
    title : str or None
        Title of the entire figure.
    """
    if expression_key == "raw":
        if adata.raw is None:
            raise ValueError("adata.raw is None but 'raw' was selected for expression_key.")
        expr_df = pd.DataFrame(adata.raw.X.toarray(), columns=adata.raw.var_names)
    else:
        expr_df = pd.DataFrame(adata.X, columns=adata.var_names)

    num_categories = len(gene_dict)
    fig, axs = plt.subplots(
        num_categories, 1,
        figsize=(min(12, max(5, max(len(v) for v in gene_dict.values()) * 0.6)), figsize_per_row * num_categories),
        constrained_layout=True
    )

    if num_categories == 1:
        axs = [axs]  # Ensure axs is iterable

    for ax, (cat_name, genes) in zip(axs, gene_dict.items()):
        selected_genes = [g for g in genes if g in expr_df.columns]
        if max_genes_per_category:
            selected_genes = selected_genes[:max_genes_per_category]

        if not selected_genes:
            ax.set_title(f"{cat_name} (No genes found)")
            ax.axis("off")
            continue

        melted = expr_df[selected_genes].melt(var_name="Gene", value_name="Expression")

        if plot_type == "violin":
            sns.violinplot(data=melted, x="Gene", y="Expression", ax=ax, inner="box", scale="width", cut=0)
        elif plot_type == "box":
            sns.boxplot(data=melted, x="Gene", y="Expression", ax=ax)
        else:
            raise ValueError(f"Unsupported plot_type: {plot_type}")

        ax.set_title(cat_name, fontsize=12)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

    if title:
        fig.suptitle(title, fontsize=14)
    plt.show()

def plot_gene_histograms(
    adata: AnnData,
    genes: List[str],
    bins: int = 50,
    subplot: bool = True,
    zscore: bool = False,
    figsize: Optional[tuple] = None,
    expression_key: Literal["X", "raw"] = "X",
    title: str = "Gene Expression Histograms",
) -> None:
    """
    Plot histograms of gene expression levels from an AnnData object.

    Parameters
    ----------
    adata : AnnData
        AnnData object with gene expression data.
    genes : List[str]
        List of gene names to plot.
    bins : int
        Number of histogram bins.
    subplot : bool
        If True, plot each gene in a subplot. If False, overlay in one plot.
    zscore : bool
        Whether to z-score the expression values before plotting.
    figsize : Optional[tuple]
        Size of the figure. Defaults to (5 * cols, 3 * rows) for subplots.
    expression_key : {"X", "raw"}
        Whether to use `adata.X` or `adata.raw.X`.
    title : str
        Main title of the plot.
    """
    # Validate genes: can pass a list of genes (some of them can be not in var_names, but at least one should be in var_names)
    available_genes = []
    for gene in genes:
        if gene in adata.var_names:
            available_genes.append(gene)
        else:
            print(f"Gene '{gene}' not found in adata.var_names.")
    if len(available_genes) == 0:
        raise ValueError("No valid genes found in adata.var_names.")
    genes = available_genes

    # Load expression matrix
    if expression_key == "X":
        expr = adata.X
    elif expression_key == "raw":
        if adata.raw is None:
            raise ValueError("adata.raw is None, cannot use 'raw' expression key.")
        expr = adata.raw.X
    else:
        raise ValueError("expression_key must be 'X' or 'raw'.")

    if issparse(expr):
        expr = expr.toarray()

    gene_indices = [adata.var_names.get_loc(gene) for gene in genes]
    data = expr[:, gene_indices]

    if zscore:
        data = StandardScaler().fit_transform(data)

    # Plot
    if subplot:
        n = len(genes)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        figsize = figsize or (5 * ncols, 3 * nrows)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = np.array(axes).reshape(-1)
        for i, gene in enumerate(genes):
            ax = axes[i]
            ax.hist(data[:, i], bins=bins, alpha=0.7, edgecolor='black')
            ax.set_title(gene)
            ax.set_xlabel("Expression" + (" (z-score)" if zscore else ""))
            ax.set_ylabel("Cell Count")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()
    else:
        plt.figure(figsize=figsize or (6, 4))
        for i, gene in enumerate(genes):
            plt.hist(data[:, i], bins=bins, alpha=0.5, label=gene, edgecolor='black')
        plt.title(title)
        plt.xlabel("Expression" + (" (z-score)" if zscore else ""))
        plt.ylabel("Cell Count")
        plt.legend()
        plt.tight_layout()
        plt.show()