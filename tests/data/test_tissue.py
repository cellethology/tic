"""
Tests for the Tissue module.

Functionalities to test:
- validate_cells_positions: Verify that the Tissue validates that all cells have consistent position dimensions.
- get_cell_by_id: Returns the correct cell for a given ID, or None if not found.
- get_biomarkers_of_all_cells: Retrieves a dictionary mapping each cell's ID to its biomarker expression.
- get_statistics_for_biomarker: Calculates the mean and standard deviation of a given biomarker across cells, and raises an error if no data is available.
- to_graph: Converts the tissue into a PyG graph using provided node feature and edge index functions.
- to_anndata: Converts the tissue into an AnnData object with the expected structure.
"""

import pytest
import numpy as np
import torch
import anndata

from tic.data.tissue import Tissue
from tic.data.cell import Cell, Biomarkers
from tests.data.mock_data import create_mock_tissue, create_mock_cell

def test_validate_cells_positions_success():
    # Create a Tissue with consistent cell positions using the mock factory.
    tissue = create_mock_tissue()
    # This should not raise an error.
    tissue.validate_cells_positions()

def test_validate_cells_positions_failure():
    # Create two cells with inconsistent position dimensions.
    cell1 = create_mock_cell(cell_id="C1", pos=(0, 0))
    cell2 = create_mock_cell(cell_id="C2", pos=(1, 1, 1))
    with pytest.raises(ValueError):
        Tissue(tissue_id="T1", cells=[cell1, cell2], position=(0, 0))

def test_get_cell_by_id():
    tissue = create_mock_tissue()
    # Using the default create_mock_tissue, cells might have IDs like "C1", "C2", etc.
    cell = tissue.get_cell_by_id("C2")
    assert cell is not None
    assert cell.cell_id == "C2"
    # Requesting a non-existent cell should return None.
    assert tissue.get_cell_by_id("NonExistent") is None

def test_get_biomarkers_of_all_cells():
    # Create a Tissue with two cells having known biomarker values.
    cell1 = create_mock_cell(cell_id="C1", biomarkers=Biomarkers(PanCK=1.0))
    cell2 = create_mock_cell(cell_id="C2", biomarkers=Biomarkers(PanCK=2.0))
    tissue = Tissue(tissue_id="T1", cells=[cell1, cell2], position=(0, 0))
    biomarkers_dict = tissue.get_biomarkers_of_all_cells("PanCK")
    assert biomarkers_dict == {"C1": 1.0, "C2": 2.0}

def test_get_statistics_for_biomarker_success():
    # Create a Tissue with two cells having known PanCK values.
    cell1 = create_mock_cell(cell_id="C1", biomarkers=Biomarkers(PanCK=1.0))
    cell2 = create_mock_cell(cell_id="C2", biomarkers=Biomarkers(PanCK=3.0))
    tissue = Tissue(tissue_id="T1", cells=[cell1, cell2], position=(0, 0))
    mean_val, std_val = tissue.get_statistics_for_biomarker("PanCK")
    # Expected mean is 2.0 and standard deviation is 1.0.
    np.testing.assert_almost_equal(mean_val, 2.0)
    np.testing.assert_almost_equal(std_val, 1.0)

def test_get_statistics_for_biomarker_failure():
    # When no cell has the given biomarker, expect a ValueError.
    cell1 = create_mock_cell(cell_id="C1", biomarkers=Biomarkers())
    cell2 = create_mock_cell(cell_id="C2", biomarkers=Biomarkers())
    tissue = Tissue(tissue_id="T1", cells=[cell1, cell2], position=(0, 0))
    with pytest.raises(ValueError):
        tissue.get_statistics_for_biomarker("NonExistentBiomarker")

def test_to_graph():
    # Create a Tissue using the mock factory.
    tissue = create_mock_tissue()
    # Define a simple node feature function: return a list containing cell.size.
    def node_feature_fn(cell):
        return [cell.size]
    # Define a simple edge index function that connects each cell with its next neighbor bidirectionally.
    def edge_index_fn(cells):
        indices = []
        n = len(cells)
        for i in range(n - 1):
            indices.append([i, i + 1])
            indices.append([i + 1, i])
        return torch.tensor(indices, dtype=torch.long)
    graph = tissue.to_graph(node_feature_fn=node_feature_fn, edge_index_fn=edge_index_fn)
    # Check that the graph has node features and an edge index.
    assert hasattr(graph, "x")
    assert hasattr(graph, "edge_index")
    # The number of node features should match the number of cells.
    assert graph.x.shape[0] == len(tissue.cells)
    # For our simple edge function, we expect 2*(n-1) edges.
    assert graph.edge_index.shape[0] == 2 * (len(tissue.cells) - 1)

def test_to_anndata():
    # Create a Tissue with two cells having known biomarker data.
    cell1 = create_mock_cell(cell_id="C1", pos=(0, 0), biomarkers=Biomarkers(PanCK=1.0, CD3=2.0))
    cell2 = create_mock_cell(cell_id="C2", pos=(1, 1), biomarkers=Biomarkers(PanCK=3.0, CD3=4.0))
    tissue = Tissue(tissue_id="T1", cells=[cell1, cell2], position=(0, 0))
    adata = tissue.to_anndata()
    # Check that the returned object is an AnnData object.
    assert isinstance(adata, anndata.AnnData)
    # Check that the expression matrix (X) has the correct shape.
    # The union of biomarkers here is expected to be ['CD3', 'PanCK'].
    assert adata.X.shape[0] == 2
    assert adata.X.shape[1] == 2
    # Ensure that the observation DataFrame has the correct cell IDs.
    assert list(adata.obs.index) == ["C1", "C2"]
    # Verify that the spatial coordinates exist in obsm and have the correct shape.
    assert "spatial" in adata.obsm
    assert adata.obsm["spatial"].shape[0] == 2
    # Verify that uns contains the tissue identifier and data level.
    assert "region_id" in adata.uns
    assert adata.uns["region_id"] == "T1"
    assert adata.uns.get("data_level") == "tissue"