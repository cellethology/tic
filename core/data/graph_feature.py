# graph_feature.py
from typing import List
import numpy as np
import torch
from scipy.spatial import Delaunay

from core.data.cell import Cell
from core.constant import ALL_BIOMARKERS, ALL_CELL_TYPES, NEIGHBOR_EDGE_CUTOFF, EDGE_TYPES

def node_feature_fn(cell: Cell) -> List[float]:
    """
    Generate node features for a cell.
    
    Features include:
      - Position coordinates (2D or 3D)
      - Cell size (scalar)
      - Cell type encoded as a one-hot vector based on ALL_CELL_TYPES
      - Biomarker expression values ordered as in ALL_BIOMARKERS
    
    :param cell: A Cell object containing attributes such as pos, size, cell_type, and biomarkers.
    :return: A list of node features as floats.
    """
    # Extract position features (assumes cell.pos is a list or tuple of floats)
    pos_features = list(cell.pos)
    
    # Cell size feature
    size_feature = [cell.size]
    
    # One-hot encode the cell type
    one_hot = [0] * len(ALL_CELL_TYPES)
    if cell.cell_type in ALL_CELL_TYPES:
        index = ALL_CELL_TYPES.index(cell.cell_type)
        one_hot[index] = 1
    
    # Collect biomarker features in the order specified by ALL_BIOMARKERS
    biomarker_features = [cell.get_biomarker(biomarker) for biomarker in ALL_BIOMARKERS]
    
    # Stack all features together
    features = pos_features + size_feature + one_hot + biomarker_features
    return features

def edge_index_fn(cells: List[Cell]) -> torch.Tensor:
    """
    Generate edge indices based on Delaunay triangulation of cell coordinates.
    
    This approach produces a sparser graph that captures local connectivity by
    connecting cells that form simplices in the Delaunay triangulation.
    
    If there are too few cells for triangulation, it falls back to a full connectivity 
    excluding self-loops.
    
    :param cells: A list of Cell objects.
    :return: A torch tensor of shape [2, num_edges] representing the edge indices.
    """
    coords = np.array([cell.pos for cell in cells])
    N = len(coords)
    # If there are not enough cells for triangulation, fallback to full connectivity
    if N < 3:
        indices = np.arange(N)
        i, j = np.meshgrid(indices, indices, indexing='ij')
        mask = i != j
        edge_index = np.vstack([i[mask], j[mask]])
        return torch.tensor(edge_index, dtype=torch.long)
    
    tri = Delaunay(coords)
    edges_set = set()
    for simplex in tri.simplices:
        # For each simplex, add edges between every pair of vertices.
        for i in range(len(simplex)):
            for j in range(i+1, len(simplex)):
                # Add both directed edges if a directed graph is desired
                edges_set.add((simplex[i], simplex[j]))
                edges_set.add((simplex[j], simplex[i]))
    if not edges_set:
        return torch.empty((2, 0), dtype=torch.long)
    
    edges = np.array(list(edges_set)).T  # Shape: [2, num_edges]
    return torch.tensor(edges, dtype=torch.long)

def edge_attr_fn(cell1: Cell, cell2: Cell, edge_cutoff: float = NEIGHBOR_EDGE_CUTOFF) -> List[float]:
    """
    Compute edge attributes between two cells.
    
    The attributes include:
      - Edge type: 0 (neighbor) if the Euclidean distance is less than NEIGHBOR_EDGE_CUTOFF,
                   otherwise 1 (distant). (Self edges, if needed, could be encoded as 2.)
      - Distance: The Euclidean distance between the two cell positions.
    
    :param cell1: The first Cell object.
    :param cell2: The second Cell object.
    :return: A list containing the edge type and the distance.
    """
    # Compute the Euclidean distance between cell positions
    distance = np.linalg.norm(np.array(cell1.pos) - np.array(cell2.pos))
    
    # Determine the edge type based on the threshold
    edge_type = EDGE_TYPES["neighbor"] if distance < edge_cutoff else EDGE_TYPES["distant"]
    
    return [edge_type, distance]