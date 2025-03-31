"""
Module: tic.data.graph_feature

Provides functions for generating node features, edge indices, and edge attributes
for graph construction from Cell objects.
"""

from typing import List, Optional
import numpy as np
import torch
from scipy.spatial import Delaunay

from tic.data.cell import Cell
from tic.constant import ALL_BIOMARKERS, ALL_CELL_TYPES, NEIGHBOR_EDGE_CUTOFF, EDGE_TYPES


def node_feature_fn(cell: Cell) -> List[float]:
    """
    Generate node features for a cell.

    The features include:
      - Position coordinates (2D or 3D)
      - Cell size (scalar)
      - One-hot encoding of cell type based on ALL_CELL_TYPES
      - Biomarker expression values ordered as in ALL_BIOMARKERS

    Parameters
    ----------
    cell : Cell
        A Cell object containing attributes such as pos, size, cell_type, and biomarkers.

    Returns
    -------
    List[float]
        A list of node features.
    """
    pos_features = list(cell.pos)
    size_feature = [cell.size]

    # One-hot encode the cell type.
    one_hot = [0] * len(ALL_CELL_TYPES)
    if cell.cell_type in ALL_CELL_TYPES:
        index = ALL_CELL_TYPES.index(cell.cell_type)
        one_hot[index] = 1

    # Collect biomarker features in the order specified by ALL_BIOMARKERS.
    biomarker_features = [cell.get_biomarker(bm) for bm in ALL_BIOMARKERS]

    features = pos_features + size_feature + one_hot + biomarker_features
    return features


def edge_index_fn(cells: List[Cell]) -> torch.Tensor:
    """
    Generate edge indices using Delaunay triangulation of cell coordinates.

    If the number of cells is fewer than 3, falls back to full connectivity (excluding self-loops).

    Parameters
    ----------
    cells : List[Cell]
        A list of Cell objects.

    Returns
    -------
    torch.Tensor
        A tensor of shape [2, num_edges] representing the edge indices.
    """
    coords = np.array([cell.pos for cell in cells])
    N = len(coords)
    if N < 3:
        indices = np.arange(N)
        i, j = np.meshgrid(indices, indices, indexing='ij')
        mask = i != j
        edge_index = np.vstack([i[mask], j[mask]])
        return torch.tensor(edge_index, dtype=torch.long)

    tri = Delaunay(coords)
    edges_set = set()
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                edges_set.add((simplex[i], simplex[j]))
                edges_set.add((simplex[j], simplex[i]))
    if not edges_set:
        return torch.empty((2, 0), dtype=torch.long)

    edges = np.array(list(edges_set)).T
    return torch.tensor(edges, dtype=torch.long)


def edge_attr_fn(
    cell1: Cell, cell2: Cell, edge_cutoff: Optional[float] = None
) -> List[float]:
    """
    Compute edge attributes between two cells.

    The attributes include:
      - Edge type: 0 ("neighbor") if the distance is below the cutoff,
                   otherwise 1 ("distant").
      - Distance: Euclidean distance between the two cell positions.

    Parameters
    ----------
    cell1 : Cell
        The first Cell object.
    cell2 : Cell
        The second Cell object.
    edge_cutoff : Optional[float]
        Distance threshold; if None, uses NEIGHBOR_EDGE_CUTOFF.

    Returns
    -------
    List[float]
        A list containing the edge type and the distance.
    """
    if edge_cutoff is None:
        edge_cutoff = NEIGHBOR_EDGE_CUTOFF
    distance = np.linalg.norm(np.array(cell1.pos) - np.array(cell2.pos))
    edge_type = EDGE_TYPES["neighbor"] if distance < edge_cutoff else EDGE_TYPES["distant"]
    return [edge_type, distance]