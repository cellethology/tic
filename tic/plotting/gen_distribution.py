# ==============================================================
# File: tic/plotting/gen_distribution.py
# ==============================================================
"""
Plot gene expression distributions for multiple gene categories.

"""
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
from typing import Mapping, Sequence, Literal, Optional
import pandas as pd

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