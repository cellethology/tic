"""
Module: tic.model.model
Defines GNN models for node representation learning and center cell biomarker expression prediction.
"""

import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    GlobalAttention,
    Set2Set,
)

from tic.model.module import GATConv, GCNConv, GINConv, GraphSAGEConv


class GNN(nn.Module):
    """
    Graph Neural Network (GNN) module.

    Parameters
    ----------
    num_layer : int
        Number of GNN layers.
    num_feat : int
        Number of input features.
    emb_dim : int
        Embedding dimension.
    node_embedding_output : str
        Output representation type ("last", "concat", "max", or "sum").
    drop_ratio : float
        Dropout rate.
    gnn_type : str
        GNN convolution type ("gin", "gcn", "graphsage", or "gat").
    """

    def __init__(
        self,
        num_layer: int = 3,
        num_feat: int = 42,
        emb_dim: int = 256,
        node_embedding_output: str = "last",
        drop_ratio: float = 0,
        gnn_type: str = "gin",
    ):
        super().__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.node_embedding_output = node_embedding_output

        # Feature embedding
        self.feat_embedding = nn.Linear(num_feat, emb_dim)
        nn.init.xavier_uniform_(self.feat_embedding.weight.data)

        # GNN layers
        self.gnns = nn.ModuleList()
        for _ in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))
            else:
                raise ValueError(f"Invalid gnn_type: {gnn_type}")

        # Batch normalization for each layer
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(emb_dim) for _ in range(num_layer)]
        )
        self.dropout = nn.Dropout(p=drop_ratio)

    def forward(self, data):
        """
        Forward pass through the network.

        Parameters
        ----------
        data : torch_geometric.data.Data or torch_geometric.data.Batch
            Input data containing 'x', 'edge_index', and 'edge_attr'.

        Returns
        -------
        torch.Tensor
            Node embeddings after applying all GNN layers.
        """
        if isinstance(data, torch_geometric.data.Batch):
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
        else:
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr

        x = self.feat_embedding(x)
        for i in range(self.num_layer):
            x = self.gnns[i](x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = torch.relu(x)
            x = self.dropout(x)

        if self.node_embedding_output == "last":
            return x
        elif self.node_embedding_output == "concat":
            # If intermediate layer outputs need to be concatenated, adjust accordingly.
            return x
        elif self.node_embedding_output == "max":
            return torch.max(x, dim=-1)[0]
        elif self.node_embedding_output == "sum":
            return torch.sum(x, dim=-1)
        return x


class GNN_pred(nn.Module):
    """
    GNN-based model for predicting the center cell's biomarker expression in a subgraph.

    Parameters
    ----------
    num_layer : int
        Number of GNN layers.
    num_feat : int
        Number of input features.
    emb_dim : int
        Embedding dimension.
    node_embedding_output : str
        Output representation type ("last", "concat", "max", or "sum").
    drop_ratio : float
        Dropout rate.
    graph_pooling : str
        Graph pooling type ("sum", "mean", "max", "attention", "set2set").
    gnn_type : str
        GNN convolution type ("gin", "gcn", "graphsage", or "gat").
    num_graph_tasks : int
        Number of graph-level prediction tasks.
    """

    def __init__(
        self,
        num_layer: int = 3,
        num_feat: int = 42,
        emb_dim: int = 256,
        node_embedding_output: str = "last",
        drop_ratio: float = 0,
        graph_pooling: str = "mean",
        gnn_type: str = "gin",
        num_graph_tasks: int = 22,
    ):
        super().__init__()
        self.gnn = GNN(
            num_layer=num_layer,
            num_feat=num_feat,
            emb_dim=emb_dim,
            node_embedding_output=node_embedding_output,
            drop_ratio=drop_ratio,
            gnn_type=gnn_type,
        )

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=nn.Linear(emb_dim, 1))
        elif graph_pooling.startswith("set2set"):
            try:
                set2set_iter = int(graph_pooling[-1])
            except ValueError:
                raise ValueError(
                    "Invalid set2set format. Use 'set2set{iter}' e.g., 'set2set3'."
                )
            self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.center_cell_pred_module = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, num_graph_tasks),
            nn.Sigmoid(),
        )

    def forward(self, data):
        """
        Forward pass for subgraph-level expression prediction.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Input data containing 'x', 'edge_index', 'edge_attr', and 'batch'.

        Returns
        -------
        tuple of torch.Tensor
            Graph-level embedding and center cell expression predictions.
        """
        node_representation = self.gnn(data)
        graph_embedding = self.pool(node_representation, data.batch)
        center_cell_pred = self.center_cell_pred_module(graph_embedding)
        return graph_embedding, center_cell_pred

    def compute_loss(self, predictions, ground_truth, mask):
        """
        Compute the loss for the center cell's biomarker expression prediction task.

        Parameters
        ----------
        predictions : torch.Tensor
            Predicted biomarker expressions.
        ground_truth : torch.Tensor
            True biomarker expression values.
        mask : torch.Tensor
            Mask indicating which nodes to consider for loss computation.

        Returns
        -------
        torch.Tensor
            Computed loss value.
        """
        masked_predictions = predictions[mask]
        masked_ground_truth = ground_truth[mask]
        loss = F.mse_loss(masked_predictions, masked_ground_truth)
        return loss