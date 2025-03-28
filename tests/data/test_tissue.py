import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from tic.data.cell import Biomarkers
from tic.data.tissue import Tissue
from tests.data.mock_data import create_mock_anndata, create_mock_cell, create_mock_tissue

# -----------------------------------------------------------------------------
# Fixtures for creating dummy AnnData and Tissue objects for testing
# -----------------------------------------------------------------------------

@pytest.fixture
def dummy_anndata():
    """
    Create a dummy AnnData object using the mock function.
    """
    return create_mock_anndata()

@pytest.fixture
def tissue_from_anndata(dummy_anndata):
    """
    Construct a Tissue object using Tissue.from_anndata() from the dummy AnnData.
    """
    tissue = Tissue.from_anndata(dummy_anndata, tissue_id="T1")
    return tissue

# -----------------------------------------------------------------------------
# Unit tests for Tissue functionality
# -----------------------------------------------------------------------------

def test_get_cell_by_id(tissue_from_anndata):
    """
    Test that get_cell_by_id() returns the correct Cell object.
    """
    tissue = tissue_from_anndata
    cell = tissue.get_cell_by_id("C2")
    assert cell is not None
    assert cell.cell_id == "C2"
    # In our mock AnnData, cell "C2" has cell_type "TypeB"
    assert cell.cell_type == "TypeB"

def test_get_biomarkers_of_all_cells(tissue_from_anndata):
    """
    Test that get_biomarkers_of_all_cells() returns a dictionary of biomarker values.
    Here, we use 'Gene1' as the biomarker.
    """
    tissue = tissue_from_anndata
    biomarker_dict = tissue.get_biomarkers_of_all_cells("Gene1")
    # Expected values from the mock AnnData:
    # C1: 1.0, C2: 3.0, C3: 5.0
    expected = {"C1": 1.0, "C2": 3.0, "C3": 5.0}
    assert biomarker_dict == expected

def test_get_statistics_for_biomarker(tissue_from_anndata):
    """
    Test that get_statistics_for_biomarker() correctly computes the mean and standard deviation.
    Also test that requesting a non-existent biomarker raises a ValueError.
    """
    tissue = tissue_from_anndata
    mean, std = tissue.get_statistics_for_biomarker("Gene2")
    # Gene2 values: 2.0, 4.0, 6.0 --> mean = 4.0, standard deviation computed using numpy.std
    np.testing.assert_almost_equal(mean, 4.0, decimal=5)
    np.testing.assert_almost_equal(std, np.std([2.0, 4.0, 6.0]), decimal=5)
    
    # Test that a non-existent biomarker raises a ValueError.
    with pytest.raises(ValueError):
        tissue.get_statistics_for_biomarker("NonExistentGene")

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

def test_to_anndata(tissue_from_anndata):
    """
    Test that Tissue.to_anndata() returns an AnnData object with the correct structure.
    """
    tissue = tissue_from_anndata
    adata = tissue.to_anndata()
    
    # Verify that uns contains the expected keys.
    assert adata.uns["data_level"] == "tissue"
    assert adata.uns["tissue_id"] == "T1"
    
    # Verify that obs has the required columns.
    for col in ["cell_id", "cell_type", "size"]:
        assert col in adata.obs.columns
    
    # Verify that obsm contains spatial coordinates.
    assert "spatial" in adata.obsm
    spatial = adata.obsm["spatial"]
    assert spatial.shape[0] == adata.obs.shape[0]

def test_to_graph(tissue_from_anndata):
    """
    Test Tissue.to_graph() by providing dummy node and edge feature functions.
    """
    tissue = tissue_from_anndata

    # Dummy node_feature_fn: returns a list [Gene1, Gene2] extracted from the cell's biomarkers.
    def node_feature_fn(cell):
        return [
            cell.biomarkers.biomarkers.get("Gene1", 0.0),
            cell.biomarkers.biomarkers.get("Gene2", 0.0)
        ]
    
    # Dummy edge_index_fn: returns a fully connected graph (including self-loops).
    def edge_index_fn(cells):
        n = len(cells)
        rows, cols = [], []
        for i in range(n):
            for j in range(n):
                rows.append(i)
                cols.append(j)
        return torch.tensor([rows, cols], dtype=torch.long)
    
    graph = tissue.to_graph(node_feature_fn, edge_index_fn)
    assert isinstance(graph, Data)
    
    # Check that node features match expected values.
    expected_features = [node_feature_fn(cell) for cell in tissue.cells]
    np.testing.assert_array_almost_equal(graph.x.numpy(), np.array(expected_features))
    
    # Check that edge_index has the correct shape: [2, n_cells * n_cells]
    n_cells = len(tissue.cells)
    assert graph.edge_index.shape == (2, n_cells * n_cells)

def test_get_microenvironment_without_graph(tissue_from_anndata):
    """
    Test that calling get_microenvironment() without a computed graph raises a ValueError.
    """
    tissue = tissue_from_anndata
    # Ensure the tissue has no precomputed graph.
    tissue.graph = None
    with pytest.raises(ValueError):
        tissue.get_microenvironment(center_cell_id="C1")