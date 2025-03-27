import pandas as pd
import pytest
import torch

from tic.data.dataset import MicroEDataset, create_microe_dataloader
from tic.data.microe import MicroE

# ---------------------------------------------------------------------------
# Fixture: Create a mock raw dataset in a temporary directory.
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_raw_dataset(tmp_path):
    """
    Creates a temporary directory with a "Raw" folder containing mock CSV files.
    The CSV files mimic a minimal raw dataset for a single region (e.g., region_id = "region1").
    """
    region_id = "region1"
    raw_dir = tmp_path / "Raw"
    raw_dir.mkdir()
    
    # Create cell_data.csv with columns: CELL_ID, X, Y
    df_coords = pd.DataFrame({
        "CELL_ID": ["C1", "C2"],
        "X": [0.0, 1.0],
        "Y": [0.0, 1.0]
    })
    df_coords.to_csv(raw_dir / f"{region_id}.cell_data.csv", index=False)
    
    # Create cell_features.csv with columns: CELL_ID, SIZE
    df_features = pd.DataFrame({
        "CELL_ID": ["C1", "C2"],
        "SIZE": [10.0, 12.0]
    })
    df_features.to_csv(raw_dir / f"{region_id}.cell_features.csv", index=False)
    
    # Create cell_types.csv with columns: CELL_ID, CELL_TYPE
    df_types = pd.DataFrame({
        "CELL_ID": ["C1", "C2"],
        "CELL_TYPE": ["Tumor", "Tumor"]
    })
    df_types.to_csv(raw_dir / f"{region_id}.cell_types.csv", index=False)
    
    # Create expression.csv with columns: CELL_ID, Biomarker1, Biomarker2
    df_expression = pd.DataFrame({
        "CELL_ID": ["C1", "C2"],
        "Biomarker1": [1.0, 2.0],
        "Biomarker2": [3.0, 4.0]
    })
    df_expression.to_csv(raw_dir / f"{region_id}.expression.csv", index=False)
    
    # Return the temporary root directory and the region id.
    return tmp_path, region_id

# ---------------------------------------------------------------------------
# Fixture: Patch graph feature functions so that node_feature_fn returns valid numbers.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_graph_features(monkeypatch):
    """
    Patch the graph feature functions used in Tissue.to_graph so that:
      - node_feature_fn returns a list containing the cell size.
      - edge_index_fn and edge_attr_fn return dummy values.
    """
    # Override node_feature_fn to return [cell.size]
    monkeypatch.setattr("tic.data.dataset.node_feature_fn", lambda cell: [cell.size])
    # Dummy edge_index: simply a bidirectional edge between two nodes (if at least two cells exist)
    monkeypatch.setattr("tic.data.dataset.edge_index_fn", lambda cells: 
                        torch.tensor([[0, 1], [1, 0]], dtype=torch.long) if len(cells) >= 2 else torch.tensor([[0], [0]], dtype=torch.long))
    # Dummy edge_attr: constant value 1.0 for each edge.
    monkeypatch.setattr("tic.data.dataset.edge_attr_fn", lambda c1, c2: 1.0)

# ---------------------------------------------------------------------------
# Unit Test: MicroEDataset length and item retrieval.
# ---------------------------------------------------------------------------
def test_microedataset_length_and_item(mock_raw_dataset, tmp_path):
    """
    Test that the MicroEDataset processes the mock raw data correctly:
      - The dataset length is > 0.
      - get_microe_item returns a MicroE instance.
    """
    root, region_id = mock_raw_dataset
    # Ensure a Cache folder exists.
    cache_dir = tmp_path / "Cache"
    cache_dir.mkdir()
    
    dataset = MicroEDataset(
        root=str(root),
        region_ids=[region_id],
        k=3,
        microe_neighbor_cutoff=200.0,
        subset_cells=False,
        center_cell_types=["Tumor"],
        pre_transform=None,
        transform=None,
        raw_to_anndata_func=None  # Use the default process_region_to_anndata
    )
    
    # Check that the dataset length is greater than 0.
    assert dataset.len() > 0
    
    # Retrieve the first microenvironment item.
    microe_item = dataset.get_microe_item(0)
    assert isinstance(microe_item, MicroE)

# ---------------------------------------------------------------------------
# Unit Test: MicroEDataset DataLoader creation.
# ---------------------------------------------------------------------------
def test_microedataloader(mock_raw_dataset, tmp_path):
    """
    Test that the dataloader created from the MicroEDataset yields batches
    containing MicroE objects.
    """
    root, region_id = mock_raw_dataset
    cache_dir = tmp_path / "Cache"
    cache_dir.mkdir()
    
    dataset = MicroEDataset(
        root=str(root),
        region_ids=[region_id],
        k=3,
        microe_neighbor_cutoff=200.0,
        subset_cells=False,
        center_cell_types=["Tumor"],
        pre_transform=None,
        transform=None,
        raw_to_anndata_func=None
    )
    dataloader = create_microe_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    batches = list(dataloader)
    assert len(batches) > 0  # At least one batch should exist.
    
    # Verify that each batch is a list of MicroE objects.
    for batch in batches:
        for item in batch:
            assert isinstance(item, MicroE)