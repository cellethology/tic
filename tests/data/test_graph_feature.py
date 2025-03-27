import numpy as np
import torch
import pytest
from tic.data.graph_feature import node_feature_fn, edge_index_fn, edge_attr_fn
from tic.data.cell import Cell, Biomarkers

# For testing, we override the constants using monkeypatch in a fixture.
@pytest.fixture(autouse=True)
def override_constants(monkeypatch):
    # Override the constants used in the module.
    monkeypatch.setattr("tic.data.graph_feature.ALL_CELL_TYPES", ["Tumor", "Stromal", "Immune"])
    monkeypatch.setattr("tic.data.graph_feature.ALL_BIOMARKERS", ["Biomarker1", "Biomarker2"])
    monkeypatch.setattr("tic.data.graph_feature.NEIGHBOR_EDGE_CUTOFF", 10.0)
    monkeypatch.setattr("tic.data.graph_feature.EDGE_TYPES", {"neighbor": 0, "distant": 1})

# --- Helper: Create a dummy Cell object ---
def create_dummy_cell(cell_id: str, pos, size: float, cell_type: str, biomarker_values: dict) -> Cell:
    """
    Create a dummy Cell instance for testing.
    """
    biomarkers = Biomarkers(**biomarker_values)
    return Cell(tissue_id="T1", cell_id=cell_id, pos=pos, size=size, cell_type=cell_type, biomarkers=biomarkers)

# --- Test node_feature_fn ---
def test_node_feature_fn():
    # Create a dummy cell with known attributes.
    cell = create_dummy_cell(
        cell_id="C1",
        pos=(1.0, 2.0),
        size=5.0,
        cell_type="Tumor",
        biomarker_values={"Biomarker1": 0.5, "Biomarker2": 1.0}
    )
    # Expected features:
    # - Position: [1.0, 2.0]
    # - Size: [5.0]
    # - One-hot for "Tumor" among ["Tumor", "Stromal", "Immune"]: [1, 0, 0]
    # - Biomarkers in order ["Biomarker1", "Biomarker2"]: [0.5, 1.0]
    expected = [1.0, 2.0, 5.0, 1, 0, 0, 0.5, 1.0]
    
    features = node_feature_fn(cell)
    # Assert that the features match the expected list.
    assert features == expected

# --- Test edge_index_fn with fewer than 3 cells (fallback to full connectivity) ---
def test_edge_index_fn_fallback():
    # Create two dummy cells.
    cell1 = create_dummy_cell("C1", (0.0, 0.0), 10.0, "Tumor", {"Biomarker1": 1.0, "Biomarker2": 2.0})
    cell2 = create_dummy_cell("C2", (1.0, 1.0), 12.0, "Tumor", {"Biomarker1": 2.0, "Biomarker2": 3.0})
    cells = [cell1, cell2]
    
    edge_index = edge_index_fn(cells)
    # With 2 cells, expect full connectivity without self-loops: edges: [[0,1],[1,0]]
    expected = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    assert torch.equal(edge_index, expected)

# --- Test edge_index_fn with 3 cells (Delaunay triangulation) ---
def test_edge_index_fn_delaunay():
    # Create three cells forming a non-collinear triangle.
    cell1 = create_dummy_cell("C1", (0.0, 0.0), 10.0, "Tumor", {"Biomarker1": 1.0, "Biomarker2": 2.0})
    cell2 = create_dummy_cell("C2", (1.0, 0.0), 12.0, "Tumor", {"Biomarker1": 2.0, "Biomarker2": 3.0})
    cell3 = create_dummy_cell("C3", (0.0, 1.0), 11.0, "Tumor", {"Biomarker1": 1.5, "Biomarker2": 2.5})
    cells = [cell1, cell2, cell3]
    
    edge_index = edge_index_fn(cells)
    # For a triangle, the Delaunay simplex is [0, 1, 2] and all directed edges among them should be present.
    expected_set = {(0,1), (1,0), (0,2), (2,0), (1,2), (2,1)}
    # Convert edge_index tensor to set of tuples.
    edges = set(tuple(edge) for edge in edge_index.t().tolist())
    assert edges == expected_set

# --- Test edge_attr_fn ---
def test_edge_attr_fn_neighbor():
    # Create two dummy cells that are close (distance < cutoff).
    cell1 = create_dummy_cell("C1", (0.0, 0.0), 10.0, "Tumor", {"Biomarker1": 1.0, "Biomarker2": 2.0})
    cell2 = create_dummy_cell("C2", (3.0, 4.0), 12.0, "Tumor", {"Biomarker1": 2.0, "Biomarker2": 3.0})
    # Euclidean distance = 5.0; since 5 < 10, edge type should be neighbor (0)
    attr = edge_attr_fn(cell1, cell2)
    expected = [0, 5.0]
    # Use np.allclose to compare the distance.
    assert attr[0] == expected[0]
    assert np.allclose(attr[1], expected[1])

def test_edge_attr_fn_distant():
    # Create two dummy cells that are far apart (distance > cutoff).
    cell1 = create_dummy_cell("C1", (0.0, 0.0), 10.0, "Tumor", {"Biomarker1": 1.0, "Biomarker2": 2.0})
    cell2 = create_dummy_cell("C2", (30.0, 40.0), 12.0, "Tumor", {"Biomarker1": 2.0, "Biomarker2": 3.0})
    # Euclidean distance = 50.0; since 50 > 10, edge type should be distant (1)
    attr = edge_attr_fn(cell1, cell2)
    expected = [1, 50.0]
    assert attr[0] == expected[0]
    assert np.allclose(attr[1], expected[1])