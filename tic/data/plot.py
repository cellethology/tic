"""
Module: plot

Provides standalone plotting functions for Tissue and MicroE objects.
These functions generate graph-based visualizations, including:
  - Tissue graph plots with nodes colored by cell type.
  - MicroE graph plots with center cells highlighted.
  - Heatmaps of neighbor biomarker matrices.
"""

from typing import List, Optional
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

from tic.constant import ALL_BIOMARKERS, ALL_CELL_TYPES
from tic.data.microe import MicroE
from tic.data.tissue import Tissue
from tic.model.feature import process_biomarker_expression


def plot_tissue_graph(tissue: Tissue, figsize=(8, 8), save_path=None) -> None:
    """
    Plot a 2D graph of a Tissue's PyG graph, coloring cells by cell type.

    Parameters
    ----------
    tissue : Tissue
        A Tissue instance with a precomputed PyG graph.
    figsize : tuple, optional
        Figure size (width, height). Default is (8, 8).
    save_path : optional
        If provided, the plot is saved to this path; otherwise, it is shown interactively.

    Raises
    ------
    ValueError
        If tissue.graph is None or if cells are missing.
    """
    if tissue.graph is None:
        raise ValueError("Tissue has no PyG graph. Please compute it before plotting.")
    if not tissue.cells:
        raise ValueError("No cells found in Tissue.")

    coords = np.array([cell.pos for cell in tissue.cells])
    if coords.shape[1] != 2:
        raise ValueError("plot_tissue_graph supports only 2D tissue data.")

    cell_types = [cell.cell_type for cell in tissue.cells]
    unique_types = list(set(cell_types))
    cmap = cm.get_cmap("tab20", len(unique_types))
    type_to_color = {t: cmap(i) for i, t in enumerate(unique_types)}

    edge_index = (
        tissue.graph.edge_index.cpu().numpy()
        if isinstance(tissue.graph.edge_index, torch.Tensor)
        else tissue.graph.edge_index
    )

    plt.figure(figsize=figsize)

    for e in range(edge_index.shape[1]):
        src, tgt = edge_index[0, e], edge_index[1, e]
        x1, y1 = coords[src]
        x2, y2 = coords[tgt]
        plt.plot([x1, x2], [y1, y2], c="gray", linewidth=0.7, alpha=0.7, zorder=1)

    node_colors = [type_to_color.get(ct, "k") for ct in cell_types]
    plt.scatter(
        coords[:, 0],
        coords[:, 1],
        s=30,
        c=node_colors,
        edgecolor="k",
        linewidth=0.5,
        zorder=2,
    )

    legend_handles = [mpatches.Patch(color=type_to_color[t], label=t) for t in unique_types]
    plt.legend(handles=legend_handles, loc="lower right", title="Cell Types")

    tissue_id = getattr(tissue, "tissue_id", "Unknown_Tissue")
    plt.title(f"Graph Plot - Tissue {tissue_id}")
    plt.axis("equal")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_microe_graph(microe: MicroE, figsize=(6, 6), save_path=None) -> None:
    """
    Plot a 2D graph of a MicroE's PyG graph, highlighting the center cell.

    Parameters
    ----------
    microe : MicroE
        A MicroE instance with a precomputed PyG graph.
    figsize : tuple, optional
        Figure size (width, height). Default is (6, 6).
    save_path : optional
        If provided, saves the plot to the specified path; otherwise, shows interactively.

    Raises
    ------
    ValueError
        If microe.graph is None or if cells are missing.
    """
    if not microe.cells:
        raise ValueError("No cells found in MicroE.")
    if microe.graph is None:
        raise ValueError("No PyG graph found in MicroE. Build or assign microe.graph first.")

    coords = np.array([cell.pos for cell in microe.cells])
    if coords.shape[1] != 2:
        raise ValueError("plot_microe_graph supports only 2D data.")

    cell_types = [cell.cell_type for cell in microe.cells]
    unique_types = list(set(cell_types))
    cmap = cm.get_cmap("tab20", len(unique_types))
    type_to_color = {t: cmap(i) for i, t in enumerate(unique_types)}

    edge_index = (
        microe.graph.edge_index.cpu().numpy()
        if isinstance(microe.graph.edge_index, torch.Tensor)
        else microe.graph.edge_index
    )

    plt.figure(figsize=figsize)

    for e in range(edge_index.shape[1]):
        src, tgt = edge_index[0, e], edge_index[1, e]
        x1, y1 = coords[src]
        x2, y2 = coords[tgt]
        plt.plot([x1, x2], [y1, y2], c="gray", linewidth=0.7, alpha=0.7, zorder=1)

    if len(coords) > 1:
        neighbor_coords = coords[1:]
        neighbor_colors = [type_to_color.get(ct, "k") for ct in cell_types[1:]]
        plt.scatter(
            neighbor_coords[:, 0],
            neighbor_coords[:, 1],
            s=30,
            c=neighbor_colors,
            edgecolor="k",
            linewidth=0.5,
            zorder=2,
        )

    center_coords = coords[0]
    plt.scatter(
        center_coords[0],
        center_coords[1],
        s=100,
        c="red",
        marker="*",
        edgecolor="k",
        linewidth=1.2,
        zorder=3,
        label=f"Center {microe.center_cell.cell_id}",
    )

    radius = 0.05 * max(coords[:, 0].ptp(), coords[:, 1].ptp())
    circle = plt.Circle(
        (center_coords[0], center_coords[1]),
        radius=radius,
        edgecolor="red",
        facecolor="none",
        linewidth=2,
        zorder=3,
    )
    plt.gca().add_patch(circle)

    plt.xlim(coords[:, 0].min(), coords[:, 0].max())
    plt.ylim(coords[:, 1].min(), coords[:, 1].max())

    legend_handles = [mpatches.Patch(color=type_to_color[t], label=t) for t in unique_types]
    plt.legend(handles=legend_handles, loc="lower right", title="Cell Types")

    title_str = (
        f"MicroE Graph\n"
        f"(Center Cell: {microe.center_cell.cell_id}, Tissue: {microe.tissue_id})"
    )
    plt.title(title_str, fontsize=12)
    plt.axis("equal")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_microe_biomarker_matrix(
    microe,
    biomarkers: List[str] = ALL_BIOMARKERS,
    cell_types: List[str] = ALL_CELL_TYPES,
    norm_method: str = "rank",
    lb: float = 0,
    ub: float = 1,
    cmap: str = "viridis",
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize the neighborhood biomarker matrix of a MicroE as a heatmap.

    For each cell type, the biomarker expression values are normalized,
    and the resulting matrix is displayed as a heatmap with appropriate labels.

    Parameters
    ----------
    microe : MicroE
        A MicroE object.
    biomarkers : List[str], optional
        List of biomarker names (default ALL_BIOMARKERS).
    cell_types : List[str], optional
        List of cell types (default ALL_CELL_TYPES).
    norm_method : str, optional
        Normalization method ('rank', 'linear', 'log', or 'raw'). Default is 'rank'.
    lb : float, optional
        Lower bound for normalization (for 'linear' or 'log').
    ub : float, optional
        Upper bound for normalization (for 'linear' or 'log').
    cmap : str, optional
        Colormap for the heatmap (default "viridis").
    figsize : tuple, optional
        Figure size (width, height). Default is (8, 6).
    save_path : str, optional
        If provided, the figure is saved to this path; otherwise, it is shown interactively.
    """
    raw_matrix = microe.get_neighborhood_biomarker_matrix(biomarkers, cell_types)
    n_types, n_biomarkers = raw_matrix.shape

    processed_rows = []
    for i in range(n_types):
        row = raw_matrix[i, :]
        processed_row = process_biomarker_expression(row, method=norm_method, lb=lb, ub=ub)
        processed_rows.append(processed_row)
    processed_matrix = np.array(processed_rows)

    row_labels = []
    for ctype in cell_types:
        ctype_count = sum(1 for cell in microe.neighbors if cell.cell_type == ctype)
        row_labels.append(f"{ctype} ({ctype_count})")

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(processed_matrix, interpolation="nearest", cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(n_biomarkers))
    ax.set_xticklabels(biomarkers, rotation=45, ha="right", fontsize=10)

    ax.set_yticks(np.arange(n_types))
    ax.set_yticklabels(row_labels, fontsize=10)

    num_neighbors = len(microe.neighbors)
    ax.text(
        0.01,
        0.01,
        f"Neighbors: {num_neighbors}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig.colorbar(cax, ax=ax)
    ax.set_xlabel("Biomarkers", fontsize=12)
    ax.set_ylabel("Cell Types", fontsize=12)

    title_str = (
        f"Neighbor Biomarker Matrix\n"
        f"(Center Cell: {microe.center_cell.cell_id}, Tissue: {microe.tissue_id})"
    )
    ax.set_title(title_str, fontsize=12)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()