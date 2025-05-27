# tic/config.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Sequence, Union

from .pseudotime.pp.clustering import ClusterMethod
from .pseudotime.pp.dimensionality import ReductionMethod


@dataclass(frozen=True, slots=True)
class PseudotimeConfig:  # noqa: D101 – docstring below
    """Hyperparameters controlling the pseudotime pipeline.
    """

    # ── Input / preprocessing ────────────────────────────────────────────────
    rep_key: str = "centre_gene"  #: key in ``adata.obsm`` used as input matrix

    # ── Dimensionality reduction ────────────────────────────────────────────
    dr_method: str | ReductionMethod = "pca"
    n_components: int = 2

    # ── Clustering ──────────────────────────────────────────────────────────
    cluster_method: str | ClusterMethod = "kmeans"
    n_clusters: int = 2

    # ── Pseudotime engine (Slingshot) ───────────────────────────────────────
    start_node: int | None = None
    epochs: int = 10  # passed to SlingshotMethod

    # ── Misc / reproducibility ──────────────────────────────────────────────
    random_state: int | None = 42

    # ── Output & metrics ────────────────────────────────────────────────────
    output_dir: Path | None = None
    monotonicity_method: Literal["spearman", "kendall"] = "spearman"
    trend_method: Literal["mann_kendall", "linear_regression"] = "mann_kendall"

    # ── Pipeline orchestration ──────────────────────────────────────────────
    step_order: Sequence[str] | None = None  #: override default execution order

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a plain‑Python representation (useful for YAML/JSON dumps)."""
        return asdict(self)

@dataclass(frozen=True, slots=True)
class GraphConfig:
    """
    Configuration for graph construction and querying.

    Attributes:
        method: Graph building method, either 'knn' or 'radius'.
        k: Number of neighbors for KNN graph.
        radius: Radius threshold for radius graph.
        metric: Distance metric to compute neighbors.
        key: Optional key under which the adjacency is stored in AnnData.obsp.
    """
    method: str | Literal["knn", "radius"] = "knn"
    k: int = 10
    radius: float | None = None
    metric: str | Literal["euclidean", "cosine"] = "euclidean"
    key: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------- Feature
@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """
    Recipe-driven micro-environment feature extraction.
    
    Attributes:
        recipe: Recipe dict or string name.
        centre_types: Types of cells to use as centres.
        graph_params: Parameters for the graph construction.
        subgraph_params: Parameters for the subgraph construction.
    """
    recipe: Union[str, Mapping[str, Mapping[str, Any]]] = "tme_default"
    centre_types: Sequence[str] | None = None
    graph_params: Mapping[str, Any] | None = None
    subgraph_params: Mapping[str, Any] | None = None
    build_graph_if_missing: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)