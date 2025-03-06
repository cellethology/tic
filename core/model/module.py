# This part of code is directly copied from the original source code:
# https://gitlab.com/enable-medicine-public/space-gm/-/blob/main/spacegm/models.py @author: zqwu
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros

NUM_NODE_TYPE = 20  # Default value
NUM_EDGE_TYPE = 4  # neighbor, distant, self


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.

    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()

        # Multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding = torch.nn.Embedding(NUM_EDGE_TYPE, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = NUM_EDGE_TYPE - 1
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # Note that the first feature of "edge_attr" should always be edge type
        # Pairwise distances are not used
        edge_embeddings = self.edge_embedding(edge_attr[:, 0].long())  
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding = torch.nn.Embedding(NUM_EDGE_TYPE, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        # assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),),
                                 dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
        deg = deg.scatter_add(0, row, edge_weight)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = NUM_EDGE_TYPE - 1
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # Note that the first feature of "edge_attr" should always be edge type
        # Pairwise distances are not used
        edge_embeddings = self.edge_embedding(edge_attr[:, 0].long())
        norm = self.norm(edge_index, x.size(0), x.dtype)
        x = self.linear(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding = torch.nn.Embedding(NUM_EDGE_TYPE, heads * emb_dim)
        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = NUM_EDGE_TYPE - 1
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # Note that the first feature of "edge_attr" should always be edge type
        # Pairwise distances are not used
        edge_embeddings = self.edge_embedding(edge_attr[:, 0].long())
        x = self.weight_linear(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return (x_j * alpha.view(-1, self.heads, 1)).view(-1, self.heads * self.emb_dim)

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.heads, self.emb_dim)
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias
        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding = torch.nn.Embedding(NUM_EDGE_TYPE, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # add self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1))
        self_loop_attr[:, 0] = NUM_EDGE_TYPE - 1
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # Note that the first feature of "edge_attr" should always be edge type
        # Pairwise distances are not used
        edge_embeddings = self.edge_embedding(edge_attr[:, 0].long())
        x = self.linear(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)