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

from tic.constant import ALL_BIOMARKERS, ALL_CELL_TYPES
from tic.data.microe import MicroE
from tic.data.tissue import Tissue
from tic.model.feature import process_biomarker_expression

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
    edge_index = (microe.graph.edge_index.cpu().numpy() 
                  if isinstance(microe.graph.edge_index, torch.Tensor) 
                  else microe.graph.edge_index)

    plt.figure(figsize=figsize)

    # Draw edges
    for e in range(edge_index.shape[1]):
        src, tgt = edge_index[0, e], edge_index[1, e]
        (x1, y1), (x2, y2) = coords[src], coords[tgt]
        plt.plot([x1, x2], [y1, y2], c='gray', linewidth=0.7, alpha=0.7, zorder=1)

    # Plot neighbors (all cells except the center cell)
    if len(coords) > 1:
        neighbor_coords = coords[1:]
        neighbor_colors = [type_to_color.get(ct, 'k') for ct in cell_types[1:]]
        plt.scatter(neighbor_coords[:, 0],
                    neighbor_coords[:, 1],
                    s=30, c=neighbor_colors, edgecolor='k', linewidth=0.5, zorder=2)
    
    # Highlight the center cell (MicroE center cell at index 0)
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
    
    title_str = (f"MicroE Graph\n"
                 f"(Center Cell: {microe.center_cell.cell_id}, Tissue: {microe.tissue_id})")
    plt.title(title_str, fontsize=12)
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    

def plot_microe_biomarker_matrix(microe, 
                                 biomarkers: list = ALL_BIOMARKERS, 
                                 cell_types: list = ALL_CELL_TYPES,
                                 norm_method: str = 'rank', 
                                 lb: float = 0, 
                                 ub: float = 1,
                                 cmap: str = 'viridis', 
                                 figsize: tuple = (8, 6),
                                 save_path: str = None):
    """
    Visualize the neighborhood biomarker matrix of a MicroE as a heatmap.
    
    Steps:
      1) Obtain the N×M matrix via microe.get_neighborhood_biomarker_matrix(), 
         where N is the number of cell types and M is the number of biomarkers.
      2) For each row (cell type), apply process_biomarker_expression to 
         normalize the row's values to [0, 1].
      3) Label each row with the cell type name plus the neighbor count for that type.
      4) Show the total neighbor count in the bottom-left corner of the plot.
      5) Display the full set of biomarkers (columns) and cell types (rows) as axis labels.

    :param microe: A MicroE object with .neighbors and get_neighborhood_biomarker_matrix().
    :param biomarkers: List of biomarker names. Defaults to ALL_BIOMARKERS.
    :param cell_types: List of cell types. Defaults to ALL_CELL_TYPES.
    :param norm_method: Normalization method ('rank', 'linear', 'log', or 'raw'). Default is 'rank'.
    :param lb: Lower bound for normalization (used by 'linear' or 'log').
    :param ub: Upper bound for normalization (used by 'linear' or 'log').
    :param cmap: Matplotlib colormap for the heatmap.
    :param figsize: Tuple specifying (width, height) of the figure.
    :param save_path: If provided, saves the figure to this path; otherwise shows it interactively.
    """
    # Get the raw biomarker matrix (N x M).
    raw_matrix = microe.get_neighborhood_biomarker_matrix(biomarkers, cell_types)
    n_types, n_biomarkers = raw_matrix.shape

    # Normalize each row to [0, 1].
    processed_rows = []
    for i in range(n_types):
        row = raw_matrix[i, :]
        processed_row = process_biomarker_expression(row, method=norm_method, lb=lb, ub=ub)
        processed_rows.append(processed_row)
    processed_matrix = np.array(processed_rows)

    # Build row labels with cell type + neighbor count.
    row_labels = []
    for ctype in cell_types:
        ctype_count = sum(1 for cell in microe.neighbors if cell.cell_type == ctype)
        row_labels.append(f"{ctype} ({ctype_count})")

    # Create the heatmap.
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(processed_matrix, interpolation='nearest', cmap=cmap, aspect='auto')

    # X-axis: biomarkers
    ax.set_xticks(np.arange(n_biomarkers))
    ax.set_xticklabels(biomarkers, rotation=45, ha='right', fontsize=10)

    # Y-axis: cell types (with counts)
    ax.set_yticks(np.arange(n_types))
    ax.set_yticklabels(row_labels, fontsize=10)

    # Show total neighbor count in bottom-left corner
    num_neighbors = len(microe.neighbors)
    ax.text(0.01, 0.01, f"Neighbors: {num_neighbors}", transform=ax.transAxes,
            fontsize=12, verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Add colorbar
    fig.colorbar(cax, ax=ax)

    ax.set_xlabel("Biomarkers", fontsize=12)
    ax.set_ylabel("Cell Types", fontsize=12)

    title_str = (f"Neighbor Biomarker Matrix\n"
                 f"(Center Cell: {microe.center_cell.cell_id}, Tissue: {microe.tissue_id})")
    ax.set_title(title_str, fontsize=12)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()