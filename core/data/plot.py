"""
plot.py

A module providing standalone plotting functions for Tissue and MicroE objects.
These functions handle graph-based visualizations,
coloring cells by cell type, and distinguishing center cells in MicroE.
"""

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

from core.data.microe import MicroE
from core.data.tissue import Tissue

###############################################################################
#                           Tissue Plotting Functions                         #
###############################################################################
def plot_tissue_graph(tissue: Tissue, figsize=(8, 8), save_path=None):
    """
    Plot the Tissue's PyG graph in 2D, coloring each cell by cell type.
    Assumes tissue.graph is a PyG Data object and each node index i 
    corresponds to tissue.cells[i].

    :param tissue: A Tissue instance with a precomputed PyG graph (Data object).
    :param figsize: (width, height) of the figure.
    :param save_path: If provided, saves the plot to this file path. Otherwise shows it.
    """

    if tissue.graph is None:
        raise ValueError("Tissue has no PyG graph. Please compute it before plotting.")
    if not tissue.cells:
        raise ValueError("No cells found in Tissue.")

    # Get the coordinates of cells
    coords = np.array([cell.pos for cell in tissue.cells])
    if coords.shape[1] != 2:
        raise ValueError("plot_tissue_graph supports only 2D tissue data.")

    # Prepare cell type information and color mapping
    cell_types = [cell.cell_type for cell in tissue.cells]
    unique_types = list(set(cell_types))
    cmap = cm.get_cmap("tab20", len(unique_types))
    type_to_color = {t: cmap(i) for i, t in enumerate(unique_types)}

    # Retrieve edge index (shape: [2, num_edges])
    edge_index = tissue.graph.edge_index.cpu().numpy() if isinstance(tissue.graph.edge_index, torch.Tensor) \
                 else tissue.graph.edge_index

    plt.figure(figsize=figsize)

    # Draw edges
    for e in range(edge_index.shape[1]):
        src, tgt = edge_index[0, e], edge_index[1, e]
        x1, y1 = coords[src]
        x2, y2 = coords[tgt]
        plt.plot([x1, x2], [y1, y2], c='gray', linewidth=0.7, alpha=0.7, zorder=1)

    # Draw nodes with their corresponding cell type colors
    node_colors = [type_to_color.get(ct, 'k') for ct in cell_types]
    plt.scatter(coords[:, 0],
                coords[:, 1],
                s=30,
                c=node_colors,
                edgecolor='k',
                linewidth=0.5,
                zorder=2)

    # Add a legend in the lower right corner showing cell types with colors
    legend_handles = [mpatches.Patch(color=type_to_color[t], label=t) for t in unique_types]
    plt.legend(handles=legend_handles, loc='lower right', title='Cell Types')

    tissue_id = getattr(tissue, "tissue_id", "Unknown_Tissue")
    plt.title(f"Graph Plot - Tissue {tissue_id}")
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

###############################################################################
#                         MicroE Plotting Functions                           #
###############################################################################
def plot_microe_graph(microe: MicroE, figsize=(6, 6), save_path=None):
    """
    Plot the MicroE's PyG graph in 2D, coloring each cell by cell type,
    and highlighting the center cell with a star marker and an additional circle.
    The x and y limits are set to the min and max coordinates of the cells.
    A legend in the lower right corner displays the cell type colors.
    
    :param microe: A MicroE instance with a precomputed PyG graph.
    :param figsize: (width, height) of the figure.
    :param save_path: If provided, saves the plot to this file path; otherwise shows it.
    """

    if not microe.cells:
        raise ValueError("No cells found in MicroE.")
    if microe.graph is None:
        raise ValueError("No PyG graph found in MicroE. Build or assign microe.graph first.")

    # Get the coordinates of all cells
    coords = np.array([cell.pos for cell in microe.cells])
    if coords.shape[1] != 2:
        raise ValueError("plot_microe_graph supports only 2D data.")

    # Prepare cell type information and color mapping
    cell_types = [cell.cell_type for cell in microe.cells]
    unique_types = list(set(cell_types))
    cmap = cm.get_cmap("tab20", len(unique_types))
    type_to_color = {t: cmap(i) for i, t in enumerate(unique_types)}

    # Convert edge_index to NumPy if it's a tensor
    edge_index = microe.graph.edge_index.cpu().numpy() if isinstance(microe.graph.edge_index, torch.Tensor) \
                 else microe.graph.edge_index

    plt.figure(figsize=figsize)

    # Draw edges
    for e in range(edge_index.shape[1]):
        src, tgt = edge_index[0, e], edge_index[1, e]
        (x1, y1), (x2, y2) = coords[src], coords[tgt]
        plt.plot([x1, x2], [y1, y2], c='gray', linewidth=0.7, alpha=0.7, zorder=1)

    # Plot neighbors (all cells except the center cell, assumed at index 0)
    if len(coords) > 1:
        neighbor_coords = coords[1:]
        neighbor_colors = [type_to_color.get(ct, 'k') for ct in cell_types[1:]]
        plt.scatter(neighbor_coords[:, 0],
                    neighbor_coords[:, 1],
                    s=30, c=neighbor_colors, edgecolor='k', linewidth=0.5, zorder=2)
    
    # Highlight the center cell (assumed to be at index 0)
    center_coords = coords[0]
    plt.scatter(center_coords[0], center_coords[1],
                s=100, c='red', marker='*', edgecolor='k', linewidth=1.2,
                zorder=3, label=f"Center {microe.center_cell.cell_id}")
    
    # Draw an additional circle around the center cell for emphasis.
    # The radius is set to 5% of the maximum range in x or y.
    radius = 0.05 * max(coords[:, 0].ptp(), coords[:, 1].ptp())
    circle = plt.Circle((center_coords[0], center_coords[1]), radius=radius,
                        edgecolor='red', facecolor='none', linewidth=2, zorder=3)
    plt.gca().add_patch(circle)
    
    # Set x and y limits to the min and max of the cell coordinates
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Create legend handles for the cell types
    legend_handles = [mpatches.Patch(color=type_to_color[t], label=t) for t in unique_types]
    plt.legend(handles=legend_handles, loc='lower right', title='Cell Types')
    
    plt.title(f"Graph Plot - MicroE Center {microe.center_cell.cell_id}")
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()