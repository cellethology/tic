"""
 tic.plotting.annotation
 ============================
 
 Helper utilities for visualising marker genes after automatic cell-type
 annotation.  This module introduces a single public function
 ``plot_marker_genes`` that generates a dotplot or heat-map of the top
 marker genes for each cell type (or any other categorical grouping) stored
 in ``adata.obs``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData

__all__ = ["plot_marker_genes"]


def _ensure_rank_genes(
    adata: AnnData,
    *,
    groupby: str,
    n_genes: int,
    method: str,
    use_raw: bool | None,
) -> str:
    """Run *rank_genes_groups* if results are missing, return the key_used."""
    key_added = f"rank_genes_{groupby}"
    if key_added not in adata.uns:
        sc.tl.rank_genes_groups(
            adata,
            groupby=groupby,
            n_genes=n_genes,
            method=method,
            key_added=key_added,
            use_raw=use_raw,
        )
    return key_added


def plot_marker_genes(
    adata: AnnData,
    *,
    groupby: str = "pred_cell_type",
    n_genes: int = 5,
    plot_type: Literal["dotplot", "heatmap"] = "dotplot",
    n_top_genes: Optional[int] = 2000,
    save_path: str | Path | None = None,
    rank_genes_groups_method: str = "t-test",
    use_raw: bool | None = None,
    show: bool = True,
    **kwargs,
) -> None:
    """Visualise the *n_genes* strongest marker genes for each ``groupby``.

    The function will *automatically* perform the following preprocessing
    steps if necessary:

    1. **Normalisation + log-transform** (``scanpy.pp.normalize_total`` and
       ``scanpy.pp.log1p``) if these steps were not performed previously.
    2. **Highly-variable gene (HVG) selection** – keeps only the top
       ``n_top_genes`` HVGs (set ``n_top_genes=None`` to skip).

    Parameters
    ----------
    adata
        Annotated single-cell data matrix.
    groupby
        The column in ``adata.obs`` that defines cell groups (default:
        "pred_cell_type").
    n_genes
        Top *n* genes per group to display.
    plot_type
        Either ``"dotplot"`` (default) or ``"heatmap"``.
    n_top_genes
        Number of highly variable genes to keep **before** differential
        expression.  If ``None``, HVG filtering is skipped.
    save_path
        Optional file path – if given, the figure is saved here (parent
        folders are created automatically).
    rank_genes_groups_method
        Differential-expression method passed to
        :pyfunc:`scanpy.tl.rank_genes_groups` when results are not already
        cached.
    use_raw
        Whether to use ``adata.raw`` for DE analysis.  ``None`` = Scanpy
        default.
    show
        Display the plot interactively.  If ``False`` and *save_path* is not
        provided, the figure is silently closed to avoid GUI windows during
        batch runs.
    **kwargs
        Forwarded to the underlying Scanpy plotting function.
    """
    if groupby not in adata.obs:
        raise KeyError(
            f"'{groupby}' not found in adata.obs. Available keys: {list(adata.obs.columns)}"
        )

    # ------------------------------------------------------------------
    # 1. Normalisation & log1p (idempotent)
    # ------------------------------------------------------------------
    if "log1p" not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # ------------------------------------------------------------------
    # 2. Highly-variable gene selection (optional)
    # ------------------------------------------------------------------
    if n_top_genes is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)

    # ------------------------------------------------------------------
    # 3. Differential expression & plotting
    # ------------------------------------------------------------------
    key_used = _ensure_rank_genes(
        adata,
        groupby=groupby,
        n_genes=n_genes,
        method=rank_genes_groups_method,
        use_raw=use_raw,
    )

    if plot_type == "dotplot":
        grid = sc.pl.rank_genes_groups_dotplot(
            adata,
            key=key_used,
            n_genes=n_genes,
            show=show,
            **kwargs,
        )
        fig = grid.fig if hasattr(grid, "fig") else grid

    elif plot_type == "heatmap":
        grid = sc.pl.rank_genes_groups_heatmap(
            adata,
            key=key_used,
            n_genes=n_genes,
            show=show,
            **kwargs,
        )
        fig = grid.fig if hasattr(grid, "fig") else grid
    else:
        raise ValueError("plot_type must be either 'dotplot' or 'heatmap'.")

    # ------------------------------------------------------------------
    # 4. Optional file output
    # ------------------------------------------------------------------
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # ------------------------------------------------------------------
    # 5. Clean-up in non-interactive mode
    # ------------------------------------------------------------------
    if (not show) and (save_path is None):
        plt.close(fig)
