# tic/graph/tl/subgraph.py
"""Multiple implementations for subgraph extraction from spatial graphs."""


from __future__ import annotations

from functools import lru_cache
from typing import Literal, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import BallTree, KDTree

from ..utils import estimate_radius, get_connectivities_key

try:
    from anndata import AnnData
except ImportError as exc:
    raise ImportError("`extract_subgraph` requires the `anndata` package.") from exc

# Optional deps
try:
    import networkx as nx  # noqa: WPS433
except ImportError:
    nx = None  # type: ignore

try:
    from torch_geometric.utils import k_hop_subgraph  # noqa: WPS433
    from torch_geometric.data import Data  # noqa: WPS433
except ImportError:
    k_hop_subgraph = None  # type: ignore
    Data = None  # type: ignore

ReturnType = Literal["indices", "networkx", "pyg"]
Strategy = Literal["knn", "radius", "k_hop_shortest", "k_hop_pyg", "bfs_python", "slice_adj"]


def _to_index(center: Union[int, str], adata: AnnData) -> int:
    """Convert obs name or int into row index in adata."""
    if isinstance(center, str):
        return adata.obs_names.get_loc(center)
    return int(center)


_COORDS_MAP: dict[int, np.ndarray] = {}

@lru_cache(maxsize=4)
def _build_trees(coords_id: int, metric: str = "euclidean") -> Tuple[BallTree, KDTree]:
    """
    Build and cache BallTree and KDTree from globally stored coords.
    """
    coords = _COORDS_MAP[coords_id]
    return BallTree(coords, metric=metric), KDTree(coords, metric=metric)


def _extract_radius(coords: np.ndarray, center_idx: int, radius: float, metric: str) -> np.ndarray:
    coords_id = id(coords)
    _COORDS_MAP[coords_id] = coords
    bt, _ = _build_trees(coords_id, metric)
    return bt.query_radius(coords[[center_idx]], r=radius)[0].astype(int)


def _extract_knn(coords: np.ndarray, center_idx: int, k: int, metric: str) -> np.ndarray:
    coords_id = id(coords)
    _COORDS_MAP[coords_id] = coords
    _, kd = _build_trees(coords_id, metric)
    k_eff = min(coords.shape[0], k + 1)
    return kd.query(coords[[center_idx]], k=k_eff, return_distance=False)[0].astype(int)

def _extract_khop_shortest(
    mat: sp.spmatrix, center_idx: int, hop: int
) -> np.ndarray:
    """Use unweighted shortest_path to get nodes ≤ hop away."""
    dist = shortest_path(csgraph=mat, directed=False,
                         unweighted=True, indices=center_idx)
    return np.where((dist != np.inf) & (dist <= hop))[0].astype(int)


def _extract_bfs_python(
    mat: sp.spmatrix, center_idx: int, hop: int
) -> np.ndarray:
    """Pure-Python BFS to collect k-hop neighbours."""
    frontier = {center_idx}
    visited = {center_idx}
    for _ in range(hop):
        next_front = set()
        for u in frontier:
            nbrs = mat.indices[mat.indptr[u] : mat.indptr[u + 1]]
            next_front.update(nbrs)
        frontier = next_front - visited
        if not frontier:
            break
        visited |= frontier
    return np.array(sorted(visited), dtype=int)


def _extract_slice_adj(
    mat: sp.spmatrix, center_idx: int, hop: int
) -> np.ndarray:
    """
    Slice adjacency matrix for nodes within hop using repeated squaring.
    
    Note: this builds mat^hop to see reachability—can be heavy for large k.
    """
    A = mat.astype(bool).astype(int).tocsr()
    M = A.copy()
    for _ in range(hop - 1):
        M = M.dot(A).astype(bool).astype(int)
    reachable = M[center_idx].nonzero()[1]
    return np.unique(np.append(reachable, center_idx)).astype(int)


def extract_subgraph(
    adata: AnnData,
    center: Union[int, str],
    *,
    strategy: Strategy = "knn",
    k: int = 6,
    radius: float | None = None,
    hop: int = 2,
    metric: str = "euclidean",
    return_type: ReturnType = "indices",
    graph_key: str | None = None,
) -> Union[np.ndarray, "nx.Graph", Data]: # type: ignore
    """
    Extract subgraph around `center` by various strategies.

    Parameters
    ----------
    adata
    center
    strategy
        'knn'             - spatial k-NN
        'radius'          - spatial radius if radius is not set, will estimate the radius-cutoff value based on the target microenv_size(default=30)
            for more details, please refer to ``tic.graph.utils.estimate_radius``
        'k_hop_shortest'  - shortest_path() BFS
        'k_hop_pyg'       - torch_geometric.k_hop_subgraph
        'bfs_python'      - pure-Python BFS
        'slice_adj'       - adjacency slicing via powers
    k
        neighbours for 'knn'
    radius
        distance for 'radius' if strategy=='radius' ; do distance cutoff for nodes
    hop
        hop count for k-hop methods
    metric
    return_type
    graph_key

    Returns
    -------
    indices: np.ndarray
        The indices of the subgraph nodes.
    nx.Graph:
        The subgraph as a networkx graph.
    Data:
        The subgraph as a torch_geometric Data object.

    Examples
    --------
    >>> from tic.graph.tl import extract_subgraph

    >>> adata = tic.graph.pp.neighbors.compute_neighbors(adata, method='voronoi')
    >>> subgraph_indices = extract_subgraph(adata, center=0, strategy='knn', k=6) # extract the subgraph of the first cell via k-nearest neighbors
    >>> subgraph_networkx = extract_subgraph(adata, center=0, strategy='radius', radius=100, return_type='networkx') # extract the subgraph of the first cell via radius
    >>> subgraph_pyg = extract_subgraph(adata, center=0, strategy='k_hop_shortest', hop=2, return_type='pyg') # extract the subgraph of the first cell via shortest path
    """
    idx = _to_index(center, adata)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)

    if strategy == "knn":
        node_idx = _extract_knn(coords, idx, k, metric)
    elif strategy == "radius":
        if radius is None:
            radius = estimate_radius(adata, target_microenv_size=30)
        node_idx = _extract_radius(coords, idx, radius, metric)
    else:
        key = get_connectivities_key(graph_key)
        if key not in adata.obsp:
            raise KeyError(f"Graph '{key}' not found; run compute_neighbors first.")
        mat = adata.obsp[key].tocsr(copy=False)
        if strategy == "k_hop_shortest":
            node_idx = _extract_khop_shortest(mat, idx, hop)
        elif strategy == "bfs_python":
            node_idx = _extract_bfs_python(mat, idx, hop)
        elif strategy == "slice_adj":
            node_idx = _extract_slice_adj(mat, idx, hop)
        elif strategy == "k_hop_pyg":
            if k_hop_subgraph is None:
                raise ImportError("install torch_geometric for 'k_hop_pyg'")
            subset, edge_idx, _, _ = k_hop_subgraph(idx, hop, mat, relabel_nodes=True)
            node_idx = subset.numpy()
        else:
            raise ValueError(f"Unknown strategy '{strategy}'")

        if radius is not None:
            dists = np.linalg.norm(coords[node_idx] - coords[idx], axis=1)
            node_idx = node_idx[dists <= radius]

    node_idx = np.unique(node_idx)

    if return_type == "indices":
        return node_idx

    if strategy in {"knn", "radius"}:
        nbrs = node_idx[node_idx != idx]
        src = np.concatenate([np.full_like(nbrs, idx), nbrs])
        dst = np.concatenate([nbrs, np.full_like(nbrs, idx)])
        edges = np.vstack([src, dst])
    else:
        sub = adata.obsp[get_connectivities_key(graph_key)][np.ix_(node_idx, node_idx)].tocsr()
        r, c = sub.nonzero()
        edges = np.vstack([r, c]).astype(int)

    if return_type == "networkx":
        if nx is None:
            raise ImportError("install networkx for 'networkx' output")
        G = nx.Graph()
        G.add_nodes_from(node_idx)
        G.add_edges_from(edges.T)
        return G

    if return_type == "pyg":
        if Data is None:
            raise ImportError("install torch_geometric for 'pyg' output")
        import torch
        return Data(edge_index=torch.tensor(edges, dtype=torch.int64), num_nodes=len(node_idx))

    raise ValueError(f"Unsupported return_type '{return_type}'")
