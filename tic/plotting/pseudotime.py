# tic/plotting/pseudotime.py
from __future__ import annotations

from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from anndata import AnnData
from typing import List, Literal, Optional

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
    bins: Optional[int] = 100,
    ax: Optional[plt.Axes] = None,
    title: str = "Biomarker Trends Along Pseudotime",
    save_path: Optional[str] = None,
) -> plt.Axes:
    """
    Plot biomarker expression trends along pseudotime with independent control over x/y-axis transformations.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing .obs[pseudotime_key] and expression data.
    pseudotime_key : str
        Key in .obs for pseudotime values.
    expression_key : str
        Key for expression matrix: "X" or key in .obsm.
    selected_biomarkers : list of str
        List of biomarkers to visualize.
    x_transform : "raw" or "bin" or "bin+normalize"
        Whether to bin pseudotime or keep raw values.
    y_transform : None | "normalize" | "smooth" | "normalize+smooth"
        Transformation for expression values.
    bins : int
        Number of bins if x_transform="bin".
    ax : plt.Axes
        Optional Axes to plot into.
    save_path : str
        If set, saves the figure. Will not show the figure.

    Examples
    --------
    >>> plot_biomarker_trends(adata, pseudotime_key='pseudotime', expression_key='X', selected_biomarkers=EMT_GENES, y_transform=None) # default usage
    """
    if pseudotime_key not in adata.obs:
        raise KeyError(f"Pseudotime key '{pseudotime_key}' not found.")

    pt = adata.obs[pseudotime_key].to_numpy()
    expr = (
        np.asarray(adata.X)
        if expression_key == "X"
        else np.asarray(adata.obsm.get(expression_key, []))
    )
    if expr.size == 0:
        raise KeyError(f"Expression key '{expression_key}' not found in AnnData.")

    var_names = list(adata.var_names) if adata.var_names is not None else []
    if selected_biomarkers:
        idxs = [var_names.index(b) for b in selected_biomarkers if b in var_names]
        if not idxs:
            raise ValueError("None of the selected biomarkers found in var_names.")
        expr = expr[:, idxs]
        names = [var_names[i] for i in idxs]
    else:
        names = var_names or [f"Var_{i}" for i in range(expr.shape[1])]

    if x_transform == "bin":
        edges = np.linspace(np.nanmin(pt), np.nanmax(pt), bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        bin_ids = np.digitize(pt, edges) - 1
        bin_ids[bin_ids == bins] = bins - 1

        binned = np.full((bins, expr.shape[1]), np.nan)
        for i in range(bins):
            sel = np.where(bin_ids == i)[0]
            if sel.size:
                binned[i] = np.nanmean(expr[sel], axis=0)

        y_data = binned
        x_axis = centers
    elif x_transform == "raw":
        sort_idx = np.argsort(pt)
        x_axis = pt[sort_idx]
        y_data = expr[sort_idx]
    elif x_transform == "bin+normalize":
        edges = np.linspace(np.nanmin(pt), np.nanmax(pt), bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        norm_centers = normalize(centers)

        bin_ids = np.digitize(pt, edges) - 1
        bin_ids[bin_ids == bins] = bins - 1

        binned = np.full((bins, expr.shape[1]), np.nan)
        for i in range(bins):
            sel = np.where(bin_ids == i)[0]
            if sel.size:
                binned[i] = np.nanmean(expr[sel], axis=0)

        y_data = binned
        x_axis = norm_centers

    else:
        raise ValueError(f"Unsupported x_transform mode: {x_transform}")

    # Apply y_transform
    # fill NaNs to avoid x-axis breaks
    for j in range(y_data.shape[1]):
        y_data[:, j] = fill_nan_with_interp(y_data[:, j])

        if y_transform:
            if "normalize" in y_transform:
                y_data[:, j] = normalize(y_data[:, j])
            if "smooth" in y_transform:
                y_data[:, j] = moving_average(y_data[:, j])

    # Plotting
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8), dpi=100)

    for j, name in enumerate(names):
        ax.plot(x_axis, y_data[:, j], label=name)

    ax.set_xlabel("Pseudotime")
    ax.set_ylabel(f"Expression {y_transform}" if y_transform else "Expression")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
        return None 
    else:
        plt.show()
        return ax