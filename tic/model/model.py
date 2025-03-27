import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from tic.model.module import GATConv, GCNConv, GINConv, GraphSAGEConv

class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): Number of GNN layers.
        num_feat (int): Number of features besides node type (e.g., biomarker expression, size).
        emb_dim (int): Dimensionality of embeddings.
        node_embedding_output (str): "last", "concat", "max", or "sum" to specify output representation.
        drop_ratio (float): Dropout rate.
        gnn_type (str): One of "gin", "gcn", "graphsage", or "gat" for specifying convolution type.
    Output:
        Node representations
    """
    def __init__(self,
                 num_layer=3,
                 num_feat=42,
                 emb_dim=256,
                 node_embedding_output="last",
                 drop_ratio=0,
                 gnn_type="gin"):
        super(GNN, self).__init__()
        
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.node_embedding_output = node_embedding_output

        # Feature embedding (for features like biomarker expressions, size)
        self.feat_embedding = torch.nn.Linear(num_feat, emb_dim)

        torch.nn.init.xavier_uniform_(self.feat_embedding.weight.data)

        # List of GNN layers based on chosen type
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))
 
        # Batch normalization for each layer
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        # Dropout layer
        self.dropout = torch.nn.Dropout(p=drop_ratio)

    def forward(self, data):
        """
        Perform forward pass through the network.
        Args:
            data (Data or Batch): The PyG Data object or Batch object containing 'x' (node features), 'edge_index', 'edge_attr', etc.
        
        Returns:
            torch.Tensor: Node embeddings after applying all GNN layers and aggregations.
        """
        # If the input is a Batch (i.e., it contains multiple graphs)
        if isinstance(data, torch_geometric.data.Batch):
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
        else:
            # For a single graph (not a batch)
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr

        # Apply feature embedding to node features
        x = self.feat_embedding(x)

        for layer in range(self.num_layer):
            # Perform convolution for the layer
            x = self.gnns[layer](x, edge_index, edge_attr)
            
            # Apply batch normalization
            x = self.batch_norms[layer](x)
            
            # Apply ReLU activation
            x = torch.relu(x)
            
            # Apply dropout for regularization
            x = self.dropout(x)

        # Depending on the output type, we can perform different aggregation methods
        if self.node_embedding_output == "last":
            return x  # Return the embeddings after the final layer
        elif self.node_embedding_output == "concat":
            return torch.cat(x, dim=-1)  # Concatenate embeddings from all layers
        elif self.node_embedding_output == "max":
            return torch.max(x, dim=-1)[0]  # Return the max of all layers
        elif self.node_embedding_output == "sum":
            return torch.sum(x, dim=-1)  # Return the sum of all layers

        return x

class GNN_pred(torch.nn.Module):
    """
    GNN-based model for predicting the center cell's biomarker expression in a subgraph.
    
    Args:
        num_layer (int): Number of GNN layers.
        num_feat (int): Number of features besides node type.
        emb_dim (int): Dimensionality of embeddings.
        node_embedding_output (str): One of "last", "concat", "max", or "sum" to specify output representation.
        drop_ratio (float): Dropout rate.
        graph_pooling (str): One of "sum", "mean", "max", "attention", "set2set" for graph pooling.
        gnn_type (str): One of "gin", "gcn", "graphsage", or "gat".
        num_node_tasks (int): Number of node-level tasks (for biomarker expression prediction).
    """
    def __init__(self,
                 num_layer=3,
                 num_feat=42,
                 emb_dim=256,
                 node_embedding_output="last",
                 drop_ratio=0,
                 graph_pooling="mean",
                 gnn_type="gin",
                 num_graph_tasks=22):
        super(GNN_pred, self).__init__()
        
        self.gnn = GNN(num_layer=num_layer,
                       num_feat=num_feat,
                       emb_dim=emb_dim,
                       node_embedding_output=node_embedding_output,
                       drop_ratio=drop_ratio,
                       gnn_type=gnn_type)

        # Graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # Center cell expression prediction module
        self.center_cell_pred_module = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LeakyReLU(),
            nn.Linear(emb_dim, num_graph_tasks),
            nn.Sigmoid()  # Sigmoid to constrain output to the range [0, 1]
        )

    def forward(self, data):
        """
        Forward pass for the subgraph-level expression prediction task (center cell prediction).
        
        Args:
            data (Data): PyG Data object containing 'x' (node features), 'edge_index', 'edge_attr', etc.
        
        Returns:
            Tensor: Graph-level embedding and center cell expression predictions (ranged between 0 and 1).
        """
        # Get node embeddings from GNN layers
        node_representation = self.gnn(data)

        # Apply graph pooling to get the graph-level embedding
        graph_embedding = self.pool(node_representation, data.batch)  # data.batch holds graph indices

        # Predict the center cell's biomarker expression based on the graph-level embedding
        center_cell_pred = self.center_cell_pred_module(graph_embedding)

        return graph_embedding, center_cell_pred

    def compute_loss(self, predictions, ground_truth, mask):
        """
        Compute the loss for the center cell's biomarker expression prediction task.
        
        Args:
            predictions (Tensor): The predicted biomarker expressions for the center cell.
            ground_truth (Tensor): The true biomarker expression values.
            mask (Tensor): Mask indicating which nodes' biomarker expressions are to be predicted.
        
        Returns:
            Tensor: The loss value.
        """
        # Only calculate loss for the masked nodes (center cell in this case)
        masked_predictions = predictions[mask]
        masked_ground_truth = ground_truth[mask]
        
        # Mean Squared Error Loss
        loss = F.mse_loss(masked_predictions, masked_ground_truth)
        return loss