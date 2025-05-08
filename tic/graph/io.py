"""Convert graphs stored in ``AnnData`` to other Python graph libraries."""
from __future__ import annotations

from typing import TYPE_CHECKING

import scipy.sparse as sp

from .utils import get_connectivities_key

if TYPE_CHECKING:  # pragma: no cover
    from anndata import AnnData
    import networkx as nx
    from torch_geometric.data import Data


# Optional dependencies are imported lazily inside functions to avoid
# hard requirements when the user only needs NumPy / SciPy.

def to_networkx(adata: "AnnData", graph_key: str | None = None):  # -> nx.Graph
    """Return the stored connectivities as a *NetworkX* graph.

    Notes
    -----
    Imports ``networkx`` only when the function is called.
    """
    import networkx as nx  # noqa: WPS433 – local import is deliberate

    key = get_connectivities_key(graph_key)
    mat: sp.spmatrix = adata.obsp[key]
    return nx.from_scipy_sparse_array(mat, create_using=nx.Graph)


def to_pyg(adata: "AnnData", graph_key: str | None = None):  # -> Data
    """Return the stored connectivities as a *PyG* ``Data`` object.

    Requires ``torch`` and ``torch_geometric``.
    """
    from torch_geometric.data import Data  # type: ignore  # noqa: WPS433
    import torch

    key = get_connectivities_key(graph_key)
    mat: sp.csr_matrix = adata.obsp[key].tocsr()
    edge_index = torch.tensor(mat.nonzero(), dtype=torch.int64)
    edge_weight = torch.tensor(mat.data, dtype=torch.float32)
    return Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=mat.shape[0])