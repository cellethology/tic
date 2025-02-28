from typing import List
import numpy as np
import torch

from core.data.cell import Cell
# Example edge feature functions

def node_feature_fn(cell: Cell):
    """
    Example function to generate node features from a cell.
    For this example, we use the cell's size and a biomarker expression (e.g., PanCK) as features.
    """
    return [cell.size] + [cell.get_biomarker("PanCK")]

def edge_index_fn(cells: List[Cell]):
    """
    Example function to generate an edge index based on cell proximity (e.g., distance threshold).
    """
    edge_index = []
    threshold_distance = 3  # Distance threshold for connectivity
    for i, cell1 in enumerate(cells):
        for j, cell2 in enumerate(cells):
            if i != j:
                distance = np.linalg.norm(np.array(cell1.pos) - np.array(cell2.pos))
                if distance < threshold_distance:
                    edge_index.append([i, j])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

def edge_attr_fn(cell1: Cell, cell2: Cell):
    """
    Example function to compute edge attributes (distance between cells).
    """
    distance = np.linalg.norm(np.array(cell1.pos) - np.array(cell2.pos))
    return distance