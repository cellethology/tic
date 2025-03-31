"""
Unit Tests for the tic/model module

This file contains unit tests for the following functionalities:
1. Feature Module:
   - process_biomarker_expression: tests ranking, linear, log, and raw methods.
   - biomarker_pretransform: tests pre-transformation of biomarker expression data.
2. Model Module:
   - GNN: tests the forward pass with dummy graph data.
   - GNN_pred: tests the forward pass and compute_loss method.
3. Transform Module:
   - mask_biomarker_expression: tests masking of biomarker expressions.
   - mask_transform: tests the masking transformation for learning tasks.
"""

import numpy as np
import torch

from torch_geometric.data import Data

from tic.model import feature, model, transform
from tic.constant import ALL_BIOMARKERS

# For testing purposes, if ALL_BIOMARKERS is empty, define a sample list.
if not ALL_BIOMARKERS:
    ALL_BIOMARKERS = ["BM1", "BM2", "BM3"]

# --- Tests for feature.py ---


def test_process_biomarker_expression_rank():
    expr = [3, 1, 2]
    processed = feature.process_biomarker_expression(expr, method="rank")
    # For ranking: the smallest value should be 0 and the largest 1.
    assert np.isclose(processed[1], 0.0)
    assert np.isclose(processed[0], 1.0)


def test_process_biomarker_expression_linear():
    expr = [0.2, 0.5, 0.8]
    processed = feature.process_biomarker_expression(expr, method="linear", lb=0, ub=1)
    np.testing.assert_allclose(processed, [(x - 0) / (1 - 0) for x in expr])


def test_process_biomarker_expression_log():
    expr = [1, 10, 100]
    processed = feature.process_biomarker_expression(expr, method="log", lb=0, ub=5)
    # Check that output values are within [0, 1]
    assert processed.min() >= 0 and processed.max() <= 1


def test_process_biomarker_expression_raw():
    expr = [0.1, 0.2, 0.3]
    processed = feature.process_biomarker_expression(expr, method="raw")
    np.testing.assert_allclose(processed, expr)


def test_biomarker_pretransform():
    # Create dummy data with 5 nodes and 10 features,
    # where the last len(ALL_BIOMARKERS)=3 columns represent biomarkers.
    num_nodes = 5
    num_features = 10
    dummy_data = torch.arange(num_nodes * num_features, dtype=torch.float).view(
        num_nodes, num_features
    )
    data = Data(x=dummy_data, edge_index=torch.tensor([[0], [0]]))
    processed_data = feature.biomarker_pretransform(data, method="rank")
    biomarkers = processed_data.x[:, -len(ALL_BIOMARKERS):]
    assert torch.all(biomarkers >= 0) and torch.all(biomarkers <= 1)


# --- Tests for model.py ---

class DummyData(Data):
    """A dummy Data class to simulate torch_geometric.data.Data with minimal attributes."""

    def __init__(self, x, edge_index, edge_attr=None, batch=None):
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr)
        self.batch = batch if batch is not None else torch.zeros(x.size(0), dtype=torch.long)


def create_dummy_graph(num_nodes=4, num_features=42):
    x = torch.rand(num_nodes, num_features)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.rand(edge_index.size(1), 4)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    return DummyData(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


def test_gnn_forward():
    dummy_data = create_dummy_graph()
    gnn = model.GNN(num_layer=2, num_feat=42, emb_dim=16, gnn_type="gin")
    output = gnn(dummy_data)
    # Output shape should be (num_nodes, emb_dim)
    assert output.shape == (dummy_data.x.size(0), 16)


def test_gnn_pred_forward_and_loss():
    dummy_data = create_dummy_graph()
    gnn_pred = model.GNN_pred(
        num_layer=2,
        num_feat=42,
        emb_dim=16,
        gnn_type="gin",
        graph_pooling="mean",
        num_graph_tasks=3,
    )
    graph_embedding, center_cell_pred = gnn_pred(dummy_data)
    num_graphs = dummy_data.batch.max().item() + 1
    assert graph_embedding.shape == (num_graphs, 16)
    assert center_cell_pred.shape == (num_graphs, 3)

    # Test compute_loss with dummy tensors.
    predictions = torch.tensor([[0.5, 0.7, 0.2], [0.6, 0.8, 0.3]])
    ground_truth = torch.tensor([[0.5, 0.7, 0.2], [0.4, 0.9, 0.3]])
    mask = torch.tensor([True, False])
    loss = gnn_pred.compute_loss(predictions, ground_truth, mask)
    expected_loss = torch.nn.functional.mse_loss(predictions[0], ground_truth[0])
    assert torch.isclose(loss, expected_loss)


# --- Tests for transform.py ---


def test_mask_biomarker_expression():
    # Create dummy Data with 3 nodes, 10 features, and define feature_indices for biomarkers.
    num_nodes = 3
    num_features = 10
    x = torch.rand(num_nodes, num_features)
    data = Data(x=x.clone(), edge_index=torch.tensor([[0], [0]]))
    data.feature_indices = {"biomarker_expression": (7, 10)}

    data = transform.mask_biomarker_expression(data, mask_ratio=0.5)
    np.testing.assert_allclose(data.y.numpy(), x[:, 7:10].numpy())
    biomarker_expr = data.x[:, 7:10]
    num_zeros = (biomarker_expr == 0).sum().item()
    assert num_zeros > 0
    assert hasattr(data, "mask")


def fake_uniform(a, b):
    return 0.2  # fixed mask ratio

def fake_rand(size, *args, **kwargs):
    # Create a tensor with some entries below 0.2 and some above 0.2.
    # For example, for a (3,3) tensor:
    return torch.tensor([[0.9, 0.1, 0.9],
                         [0.9, 0.9, 0.1],
                         [0.1, 0.9, 0.9]])

def test_mask_transform(monkeypatch):
    import torch
    from torch_geometric.data import Data
    from tic.model import transform
    import numpy as np

    # Override ALL_BIOMARKERS for the test.
    monkeypatch.setattr(transform, "ALL_BIOMARKERS", ["BM1", "BM2", "BM3"])
    # Force a fixed mask ratio.
    monkeypatch.setattr(transform, "random", type("FakeRandom", (), {"uniform": fake_uniform}))
    # Force a deterministic random tensor for masking.
    monkeypatch.setattr(transform, "torch", torch)  # ensure torch is available
    monkeypatch.setattr(torch, "rand", fake_rand)

    # Create dummy Data with 3 nodes and 10 features.
    num_nodes = 3
    num_features = 10
    x = torch.rand(num_nodes, num_features)
    data = Data(x=x.clone(), edge_index=torch.tensor([[0], [0]]))
    transformed_data = transform.mask_transform(data)

    np.testing.assert_allclose(
        transformed_data.y.numpy(), x[0, -3:].numpy()
    )
    # Now the masked part should be different from the original.
    assert not torch.allclose(
        transformed_data.x[:, -3:],
        x[:, -3:]
    )