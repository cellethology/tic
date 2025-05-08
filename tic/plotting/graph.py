# tic/plotting/graph.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Union
from matplotlib.collections import LineCollection

from ..graph.utils import get_connectivities_key

try:
    from anndata import AnnData
except ImportError as exc:  # pragma: no cover
    raise ImportError("plot_graph requires `anndata`. Install via pip.") from exc


def plot_graph(
    adata: AnnData,
    *,
    indices: Optional[Sequence[int]] = None,
    color_by: str = "cell_type",
    graph_key: Optional[str] = None,
    title: str = "Graph",
    plot_edges: bool = True,
    point_size: float = 1,
    legend_markerscale: float = 4.0,
    ax: Optional[plt.Axes] = None,
    palette: dict[Union[str, int], str] | None = None,
    save_path: str | None = None,
) -> plt.Axes:
    """
    Scatter cells in 2D spatial coords, optionally with adjacency edges.

    Parameters
    ----------
    adata : AnnData
        AnnData object with .obsm['spatial'] and optionally .obsp[connectivities].
    indices : Optional[Sequence[int]]
        Which cells to show; defaults to all.
    color_by : str
        Key in .obs to color cells.
    graph_key : Optional[str]
        Key for adjacency matrix in .obsp.
        default: 'connectivities' see: tic.graph.pp.neighbors.compute_neighbors
    plot_edges : bool
        Whether to draw graph edges.
    ax : Optional[plt.Axes]
        Axes to draw into; created if None.
    palette : dict[Union[str, int], str] | None
        Categorical color mapping.
    show : bool
        Whether to show the plot.

    Returns
    -------
    plt.Axes
        The axis containing the plot.

    Examples
    --------
    >>> from tic.plotting import plot_graph
    >>> plot_graph(adata) # plot the full graph
    >>> plot_graph(adata, plot_edges=False) # plot the full graph without edges
    >>> plot_graph(adata, indices=[0, 1, 2]) # plot the subgraph of the first 3 cells
    """
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    sel = np.array(indices) if indices is not None else np.arange(adata.n_obs)

    # Dynamic figsize based on data ratio
    if ax is None:
        x0, x1 = coords[:, 0].min(), coords[:, 0].max()
        y0, y1 = coords[:, 1].min(), coords[:, 1].max()
        width = x1 - x0
        height = y1 - y0
        ratio = height / width if width != 0 else 1.0
        fig_width = 10
        fig_height = fig_width * ratio
        _, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

    key = get_connectivities_key(graph_key)

    # Draw edges
    if plot_edges and key in adata.obsp:
        mat = adata.obsp[key].tocoo()
        mask_uv = mat.row < mat.col
        rows = mat.row[mask_uv]
        cols = mat.col[mask_uv]
        mask_sel = np.isin(rows, sel) & np.isin(cols, sel)
        rows = rows[mask_sel]
        cols = cols[mask_sel]
        segments = [
            ((coords[u, 0], coords[u, 1]), (coords[v, 0], coords[v, 1]))
            for u, v in zip(rows, cols)
        ]
        lc = LineCollection(
            segments,
            colors='gray',
            linewidths=0.5,
            alpha=0.5,
            zorder=1,
        )
        ax.add_collection(lc)

    # Draw points
    obs = adata.obs.get(color_by, None)
    if obs is None:
        ax.scatter(
            coords[sel, 0], coords[sel, 1],
            s=point_size, alpha=0.8, zorder=2
        )
    else:
        cats = obs.astype('category')
        default_cols = plt.cm.tab20.colors
        palette = palette or {
            cat: default_cols[i % len(default_cols)]
            for i, cat in enumerate(cats.cat.categories)
        }
        for cat, col in palette.items():
            mask_cat = (obs.values == cat) & np.isin(np.arange(adata.n_obs), sel)
            ax.scatter(
                coords[mask_cat, 0], coords[mask_cat, 1],
                s=point_size, label=str(cat), color=col,
                alpha=0.8, zorder=2
            )
        legend = ax.legend(
            title=color_by,
            bbox_to_anchor=(1.05, 1), loc='upper left',
            markerscale=legend_markerscale, fontsize='small', title_fontsize='small',
            handlelength=1, borderpad=0.3
        )
        legend._legend_box.align = 'left'

    # Axis labeling
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # title = "Sub-graph" if indices is not None else "Full graph"
    # if not plot_edges:
    #     title += " (no edges)"
    ax.set_title(title)

    # Axis limits with padding
    x0, x1 = coords[sel, 0].min(), coords[sel, 0].max()
    y0, y1 = coords[sel, 1].min(), coords[sel, 1].max()
    pad_x = (x1 - x0) * 0.05
    pad_y = (y1 - y0) * 0.05
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)

    ax.set_aspect("equal")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    return ax

def plot_spot(
    adata: AnnData,
    spot_id: Union[str, Sequence[str]],
    *,
    key: str = "spot_id",
    ax: Optional[plt.Axes] = None,
    title: str | None = None,
    palette: Sequence[str] = ("red", "blue", "green", "orange", "purple", "brown"),
    save_path: str | None = None,
) -> plt.Axes:
    """
    Visualize the spatial coordinates of cells in one or more specified spots.

    Parameters
    ----------
    adata : AnnData
        Cell-level AnnData object after spot assignment.
    spot_id : str or list of str
        The spot ID(s) to visualize.
    key : str, default "spot_id"
        The column in `adata.obs` indicating spot membership.
    ax : plt.Axes, optional
        Existing matplotlib axis to plot on.
    show : bool, default True
        Whether to call `plt.show()` after plotting.
    title : str, optional
        Title for the plot.
    palette : list of str
        Colors used for multiple spots. Repeats if more spots than colors.
    """
    if key not in adata.obs:
        raise KeyError(f"{key!r} not found in adata.obs.")
    if "spatial" not in adata.obsm:
        raise KeyError("`adata.obsm['spatial']` is required for plotting.")

    coords = adata.obsm["spatial"]
    spot_labels = adata.obs[key].astype(str)

    if isinstance(spot_id, str):
        spot_ids = [spot_id]
    else:
        spot_ids = list(map(str, spot_id))

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Plot all cells in gray
    ax.scatter(coords[:, 0], coords[:, 1], s=5, color="lightgray", label="all cells")

    for i, sid in enumerate(spot_ids):
        mask = spot_labels == sid
        color = palette[i % len(palette)]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=10,
            label=f"spot {sid}",
            color=color,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title or f"Cells in Spot(s): {', '.join(spot_ids)}")
    ax.legend()
    ax.axis("equal")

    if save_path:
        plt.savefig(save_path)

    return ax

    
def plot_spot_inner_cells(
    adata: AnnData,
    spot_id: str,
    *,
    key: str = "spot_id",
    cell_type_key: Optional[str] = "cell_type",
    ax: Optional[plt.Axes] = None,
    title: str | None = None,
    palette: Sequence[str] = ("tab10",), 
    save_path: str | None = None,
) -> plt.Axes:
    """
    Visualize only the spatial coordinates of cells within a specified spot, colored by cell type.

    Parameters
    ----------
    adata : AnnData
        Cell-level AnnData object after spot assignment.
    spot_id : str
        The spot ID to visualize.
    key : str, default "spot_id"
        The column in `adata.obs` indicating spot membership.
    cell_type_key : str or None, default "cell_type"
        The column in `adata.obs` indicating cell types (color by this obs column if not None).
    ax : plt.Axes, optional
        Existing matplotlib axis to plot on.
    show : bool, default True
        Whether to call `plt.show()` after plotting.
    title : str, optional
        Title for the plot.
    palette : tuple of str, default ("tab10",)
        Color map name or a list of colors. If a tuple with one element, use as cmap; else as color list.
    save_path : str, optional
        If given, save the plot to this path.
    """
    if key not in adata.obs:
        raise KeyError(f"{key!r} not found in adata.obs.")
    if "spatial" not in adata.obsm:
        raise KeyError("`adata.obsm['spatial']` is required for plotting.")

    coords = adata.obsm["spatial"]
    spot_labels = adata.obs[key].astype(str)
    mask = spot_labels == str(spot_id)

    if not mask.any():
        raise ValueError(f"No cells found for spot_id {spot_id!r}.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    plot_coords = coords[mask]

    if cell_type_key is not None and cell_type_key in adata.obs:
        cell_types = adata.obs[cell_type_key][mask].astype(str)
        unique_types = cell_types.unique()
        n_types = len(unique_types)

        # 获取色板
        if len(palette) == 1:
            # 使用matplotlib colormap名
            cmap = plt.get_cmap(palette[0])
            colors = [cmap(i / max(1, n_types - 1)) for i in range(n_types)]
        else:
            colors = list(palette) * (n_types // len(palette) + 1)
            colors = colors[:n_types]

        type_color_dict = dict(zip(unique_types, colors))

        for ct in unique_types:
            ct_mask = cell_types == ct
            ax.scatter(
                plot_coords[ct_mask, 0],
                plot_coords[ct_mask, 1],
                s=12,
                color=type_color_dict[ct],
                label=str(ct),
                alpha=0.9,
            )
    else:
        # 未指定 cell_type_key 或该key不存在
        ax.scatter(
            plot_coords[:, 0],
            plot_coords[:, 1],
            s=12,
            color=palette[0] if palette else "red",
            label=f"spot {spot_id}",
            alpha=0.9,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title or f"Cells in Spot: {spot_id}")
    ax.legend(markerscale=1.5)
    ax.axis("equal")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    return ax
