"""
Tests for the MicroE module.

Functionalities to test:
- Correct retrieval of center cell and neighbors.
- Conversion to a graph using provided node feature and edge index functions.
- Generation of neighborhood biomarker matrix.
"""

import numpy as np
import torch
import pytest
from tic.data.microe import MicroE
from tic.data.cell import Cell, Biomarkers
from tests.data.mock_data import create_mock_cell, create_mock_microe

# Simple mock functions for graph conversion
def mock_node_feature_fn(cell):
    return cell.size  # For example, using cell size as a feature

def mock_edge_index_fn(cells):
    # Connect each consecutive pair of cells with bidirectional edges.
    indices = []
    n = len(cells)
    for i in range(n - 1):
        indices.append([i, i + 1])
        indices.append([i + 1, i])
    return torch.tensor(indices, dtype=torch.long)

def test_microe_get_center_and_neighbors():
    microe = create_mock_microe()
    assert microe.get_center_cell().cell_id == "Center"
    neighbor_ids = [cell.cell_id for cell in microe.get_neighbors()]
    assert "Neighbor1" in neighbor_ids and "Neighbor2" in neighbor_ids

def test_microe_to_graph():
    microe = create_mock_microe()
    graph = microe.to_graph(node_feature_fn=mock_node_feature_fn, edge_index_fn=mock_edge_index_fn)
    # Check that graph is a PyG Data object with proper node features and edge index dimensions.
    assert hasattr(graph, "x")
    assert graph.x.shape[0] == len(microe.cells)
    assert graph.edge_index.ndim == 2

def test_microe_get_neighborhood_biomarker_matrix():
    # Create cells with distinct biomarker values
    b1 = Biomarkers(PanCK=1.0, CD3=2.0)
    b2 = Biomarkers(PanCK=1.5, CD3=2.5)
    center = create_mock_cell(cell_id="Center", cell_type="Tumor", biomarkers=b1)
    neighbor1 = create_mock_cell(cell_id="Neighbor1", cell_type="T_cell", biomarkers=b1)
    neighbor2 = create_mock_cell(cell_id="Neighbor2", cell_type="T_cell", biomarkers=b2)
    microe = MicroE(center, [neighbor1, neighbor2], tissue_id="T1")
    
    # Assume that for testing, we consider cell types ['T_cell', 'Tumor'] and biomarkers ['PanCK', 'CD3'].
    matrix = microe.get_neighborhood_biomarker_matrix(biomarkers=['PanCK', 'CD3'], cell_types=['T_cell', 'Tumor'])
    
    # For T_cell: average of PanCK values (1.0 and 1.5) = 1.25, CD3 values (2.0 and 2.5) = 2.25.
    np.testing.assert_almost_equal(matrix[0, 0], 1.25)
    np.testing.assert_almost_equal(matrix[0, 1], 2.25)
    # For Tumor: center cell is excluded, so expect NaN.
    assert np.isnan(matrix[1, 0])
    assert np.isnan(matrix[1, 1])