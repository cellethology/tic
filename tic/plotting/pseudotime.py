# tic/plotting/pseudotime.py
from __future__ import annotations

from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from anndata import AnnData
from typing import List, Literal, Optional
from scipy.stats import spearmanr

from .utils import normalize, moving_average, fill_nan_with_interp


def scatter_embedding(
    emb: np.ndarray,
    clusters: np.ndarray,
    pseudotime: np.ndarray,
    *,
    kind: str = "pseudotime",
    ax: plt.Axes | None = None,
    palette: dict[int, str] | None = None,
    remove_outliers: bool = True,
    iqr_threshold: float = 1.5,
    save_path: Optional[str] = None,
) -> plt.Axes:
    """Shared scatter plot used by the wrapper (keeps plotting out of core logic)."""
    
    assert emb.shape[0] == clusters.shape[0] == pseudotime.shape[0], "Shape mismatch in inputs"

    # Step 1: Remove outliers based on IQR
    if remove_outliers:
        q1 = np.percentile(emb, 25, axis=0)
        q3 = np.percentile(emb, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - iqr_threshold * iqr
        upper_bound = q3 + iqr_threshold * iqr
        mask = np.all((emb >= lower_bound) & (emb <= upper_bound), axis=1)
        emb = emb[mask]
        clusters = clusters[mask]
        pseudotime = pseudotime[mask]

    # Step 2: Plot
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5), dpi=110)

    if kind == "pseudotime":
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=8, c=pseudotime, cmap="viridis")
        plt.colorbar(sc, ax=ax, label="Pseudotime")
    elif kind == "cluster":
        cats = np.unique(clusters)
        if palette is None:
            palette = {int(c): cm.tab20(i / len(cats)) for i, c in enumerate(cats)}
        for c in cats:
            mask = clusters == c
            ax.scatter(emb[mask, 0], emb[mask, 1], s=10, color=palette[int(c)], label=str(c))
        ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        raise ValueError("kind must be 'pseudotime' or 'cluster'.")

    # Step 3: Make axes visually square
    x_min, x_max = np.min(emb[:, 0]), np.max(emb[:, 0])
    y_min, y_max = np.min(emb[:, 1]), np.max(emb[:, 1])
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    max_range = max(x_max - x_min, y_max - y_min) / 2
    ax.set_xlim(x_center - max_range, x_center + max_range)
    ax.set_ylim(y_center - max_range, y_center + max_range)

    ax.set(title=kind.capitalize(), xlabel="Dim 1", ylabel="Dim 2", aspect="equal")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    return ax

def plot_biomarker_trends(
    adata: AnnData,
    pseudotime_key: str = "pseudotime",
    expression_key: str = "X",
    selected_biomarkers: Optional[List[str]] = None,
    x_transform: Literal["raw", "bin", "bin+normalize"] = "bin",
    y_transform: Optional[Literal["normalize", "smooth", "normalize+smooth"]] = None,
    bins: int = 100,
    show_scatter: bool = False,
    scatter_frac: float = 0.3,
    conf_interval: bool = False,
    facet: bool = False,
    facet_ncols: int = 2,
    window: int = 5,
    ax: Optional[plt.Axes] = None,
    show_spearman: bool = False,
    title: str = "Biomarker Trends Along Pseudotime",
    save_path: Optional[str] = None,
) -> Optional[plt.Axes]:
    """
    Plot biomarker expression trends along pseudotime with options for scatter overlay,
    confidence interval shading, and facet (subplots) mode.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing .obs[pseudotime_key] and expression data in .X or .obsm[expression_key].
    pseudotime_key : str
        Key in .obs for pseudotime values.
    expression_key : str
        Key for expression matrix: "X" or key in .obsm.
    selected_biomarkers : list of str, optional
        List of biomarkers (variable names) to visualize. If None, uses all variables.
    x_transform : {"raw", "bin", "bin+normalize"}
        How to treat pseudotime on x-axis:
        - "raw": use raw pseudotime values (sorted).
        - "bin": bin pseudotime into equal-width intervals and take mean per bin.
        - "bin+normalize": same as "bin", then normalize bin centers to [0,1].
    y_transform : {None, "normalize", "smooth", "normalize+smooth"}, optional
        How to transform expression values per gene across bins or raw:
        - "normalize": scale each gene's trend to [0,1].
        - "smooth": apply moving average (window size controlled by `window`).
        - "normalize+smooth": apply normalize then smooth.
    bins : int
        Number of bins to use if x_transform starts with "bin". Default 100.
    show_scatter : bool
        Whether to overlay raw expression scatter points (subsampled) beneath the trend lines.
    scatter_frac : float
        Fraction (0<scatter_frac<=1) or absolute number of cells to subsample for scatter. Default 0.3 (30%).
    conf_interval : bool
        Whether to compute and display confidence interval (SEM) shading around mean trend.
        Only applicable when x_transform is "bin" or "bin+normalize".
    facet : bool
        If True, draw each gene in its own subplot (faceting) instead of overlaying all in one Axes.
    facet_ncols : int
        Number of columns in the facet grid. Only used if facet=True.
    window : int
        Window size for moving average when y_transform includes "smooth". Default 5.
    ax : plt.Axes, optional
        Axes to plot into (only used if facet=False). If None, a new figure/axes is created.
    show_spearman: bool
        Whether to show Spearman correlation between biomarker and pseudotime.
    title : str
        Title for the overall figure (if facet=False) or for the first subplot.
    save_path : str, optional
        If set, save the resulting figure to this path and close it, returning None.

    Returns
    -------
    ax : plt.Axes or None
        If facet=False and save_path is None, returns the Axes containing the plot.
        Otherwise, returns None.

    Raises
    ------
    KeyError
        If pseudotime_key not in adata.obs or expression_key not found.
    ValueError
        If selected_biomarkers are not found or if x_transform/y_transform modes are unsupported.

    Examples
    --------
    >>> plot_biomarker_trends(
    ...     adata,
    ...     pseudotime_key="pseudotime",
    ...     selected_biomarkers=["VIM", "CDH1", "SNAI1"],
    ...     x_transform="bin",
    ...     y_transform="normalize+smooth",
    ...     bins=80,
    ...     show_scatter=True,
    ...     scatter_frac=0.2,
    ...     conf_interval=True,
    ...     facet=False,
    ... )
    """
    # 1. Validate pseudotime
    if pseudotime_key not in adata.obs:
        raise KeyError(f"Pseudotime key '{pseudotime_key}' not found in adata.obs.")
    pt = adata.obs[pseudotime_key].to_numpy()

    # 2. Extract expression matrix
    if expression_key == "X":
        expr_full = np.asarray(adata.X)
    else:
        expr_full = np.asarray(adata.obsm.get(expression_key, []))
    if expr_full.size == 0:
        raise KeyError(f"Expression key '{expression_key}' not found in AnnData.")

    # 3. Select biomarkers / variables
    var_names = list(adata.var_names) if adata.var_names is not None else []
    if selected_biomarkers:
        idxs = [var_names.index(b) for b in selected_biomarkers if b in var_names]
        if not idxs:
            raise ValueError("None of the selected_biomarkers found in var_names.")
        expr = expr_full[:, idxs]
        names = [var_names[i] for i in idxs]
    else:
        expr = expr_full
        names = var_names if var_names else [f"Var_{i}" for i in range(expr.shape[1])]

    n_genes = expr.shape[1]
    n_cells = expr.shape[0]

    # 4. Color map for plotting
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(n_genes)]

    # 5. Compute binning if needed
    if x_transform in ("bin", "bin+normalize"):
        # Define bin edges and centers
        edges = np.linspace(np.nanmin(pt), np.nanmax(pt), bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        if x_transform == "bin+normalize":
            x_axis = normalize(centers)
        else:
            x_axis = centers

        # Assign each cell to a bin
        bin_ids = np.digitize(pt, edges) - 1
        bin_ids[bin_ids == bins] = bins - 1

        # Initialize matrices: mean, SEM, and count
        mean_vals = np.full((bins, n_genes), np.nan)
        sem_vals = np.full((bins, n_genes), np.nan)
        counts = np.zeros(bins, dtype=int)
        for i in range(bins):
            sel = np.where(bin_ids == i)[0]
            counts[i] = sel.size
            if sel.size > 0:
                bin_expr = expr[sel, :]  # shape: (n_sel, n_genes)
                mean_vals[i, :] = np.nanmean(bin_expr, axis=0)
                if conf_interval:
                    sem_vals[i, :] = np.nanstd(bin_expr, axis=0) / np.sqrt(sel.size)
        # y_data starts as mean_vals
        y_data = mean_vals.copy()

    elif x_transform == "raw":
        # Sort by pseudotime
        sort_idx = np.argsort(pt)
        x_axis = pt[sort_idx]
        y_data = expr[sort_idx, :]
        # SEM not applicable for raw mode
        mean_vals = None
        sem_vals = None
    else:
        raise ValueError(f"Unsupported x_transform mode: {x_transform}")

    # 6. Apply y_transform (normalize and/or smooth) to y_data
    if x_transform in ("bin", "bin+normalize"):
        # For each gene, fill NaN then normalize/smooth
        for j in range(n_genes):
            y_data[:, j] = fill_nan_with_interp(y_data[:, j])
            if y_transform:
                if "normalize" in y_transform:
                    y_data[:, j] = normalize(y_data[:, j])
                if "smooth" in y_transform:
                    y_data[:, j] = moving_average(y_data[:, j], window=window)
            # Also transform sem_vals if confidence interval shading is requested
            if conf_interval and sem_vals is not None and "normalize" in (y_transform or ""):
                # If normalized, scale SEM by same factor: sem / (max-min)
                gene_vals = mean_vals[:, j]
                min_v = np.nanmin(gene_vals)
                max_v = np.nanmax(gene_vals)
                if max_v > min_v:
                    sem_vals[:, j] = sem_vals[:, j] / (max_v - min_v) + 1e-8
                else:
                    sem_vals[:, j] = sem_vals[:, j]
            if conf_interval and sem_vals is not None and "smooth" in (y_transform or ""):
                sem_vals[:, j] = moving_average(sem_vals[:, j], window=window)
    elif x_transform == "raw":
        # For raw, only fill NaN in y_data (if any)
        for j in range(n_genes):
            y_data[:, j] = fill_nan_with_interp(y_data[:, j])
            if y_transform:
                if "normalize" in y_transform:
                    y_data[:, j] = normalize(y_data[:, j])
                if "smooth" in y_transform:
                    y_data[:, j] = moving_average(y_data[:, j], window=window)

    # 7. Prepare for plotting
    if facet:
        # Determine subplot grid dimensions
        ncols = facet_ncols
        nrows = int(np.ceil(n_genes / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        axes = axes.flatten()
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

    # 8. Plotting
    for j, name in enumerate(names):
        color = colors[j]

        if facet:
            curr_ax = axes[j]
        else:
            curr_ax = ax

        # 8.1 Scatter overlay (only for raw or bin modes)
        if show_scatter:
            sample_size = (
                int(n_cells * scatter_frac)
                if scatter_frac <= 1
                else int(scatter_frac)
            )
            sample_idx = np.random.choice(n_cells, size=min(n_cells, sample_size), replace=False)

            scatter_alpha = 0.1 if not facet else 0.2
            scatter_size = 4 if not facet else 5

            # Use secondary y-axis for non-facet to decouple line/scatter
            if not facet:
                scatter_ax = curr_ax.twinx()
                scatter_ax.scatter(
                    pt[sample_idx],
                    expr[sample_idx, j],
                    s=scatter_size,
                    alpha=scatter_alpha,
                    color=color,
                    zorder=1,
                )
                scatter_ax.get_yaxis().set_visible(False)
            else:
                curr_ax.scatter(
                    pt[sample_idx],
                    expr[sample_idx, j],
                    s=scatter_size,
                    alpha=scatter_alpha,
                    color=color,
                    zorder=1,
                )

        # 8.2 Plot mean trend line
        curr_ax.plot(
            x_axis,
            y_data[:, j],
            label=name if not facet else None,
            color=color,
            linewidth=2 if not facet else 1.5,
            zorder=2,
        )

        # 8.3 Confidence interval shading
        if conf_interval and x_transform in ("bin", "bin+normalize"):
            lower = y_data[:, j] - sem_vals[:, j]
            upper = y_data[:, j] + sem_vals[:, j]
            curr_ax.fill_between(
                x_axis,
                lower,
                upper,
                color=color,
                alpha=0.3,
                zorder=1.5,
            )

        # 8.4 Aesthetics for each subplot
        if show_spearman:
            try:
                rho, _ = spearmanr(pt, expr[:, j])
                title_str = f"{name} (ρ={rho:.2f})"
            except Exception:
                title_str = f"{name} (ρ=NaN)"
        else:
            title_str = name
        curr_ax.set_title(title_str)

        if facet:
            curr_ax.set_ylabel("Expression" + (f" ({y_transform})" if y_transform else ""))
            curr_ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.3)
            idx_row = j // facet_ncols
            if idx_row == nrows - 1:
                curr_ax.set_xlabel("Pseudotime" + (" (normalized)" if x_transform == "bin+normalize" else ""))
        else:
            curr_ax.set_xlabel("Pseudotime" + (" (normalized)" if x_transform == "bin+normalize" else ""))
            curr_ax.set_ylabel("Expression" + (f" ({y_transform})" if y_transform else ""))
            curr_ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.3)

    # 9. Final adjustments: legend, layout, title
    if facet:
        # Remove any unused subplots
        for idx in range(n_genes, len(axes)):
            fig.delaxes(axes[idx])
        # Optionally set a common title
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
    else:
        if show_spearman:
            legend_labels = []
            for j, name in enumerate(names):
                try:
                    rho, _ = spearmanr(pt, expr[:, j])
                    label = f"{name} (ρ={rho:.2f})"
                except Exception:
                    label = f"{name} (ρ=NaN)"
                legend_labels.append(label)
            ax.legend(legend_labels, ncol=1, fontsize="medium",
                    bbox_to_anchor=(1.03, 1), loc="upper left", borderaxespad=0.0)
        else:
            ax.legend(
                ncol=1,
                fontsize="medium",
                bbox_to_anchor=(1.03, 1),
                loc="upper left",
                borderaxespad=0.0,
            )
        ax.set_title(title)
        plt.tight_layout()

    # 10. Save or show
    if save_path:
        if facet:
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
        else:
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
        return None
    else:
        if facet:
            plt.show()
            return None
        else:
            plt.show()
            return ax