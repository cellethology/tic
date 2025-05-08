
# tic.wrappers Module

This module provides high-level wrappers for various functionalities in TIC, including graph construction, feature extraction, pseudotime inference, and causal analysis.

## Structure

```
tic/wrappers/
├── base.py               # Abstract base class for wrappers (BaseWrapper)
├── graph.py              # Graph construction and subgraph query
├── feature.py            # Feature extraction using named recipes
├── pseudotime.py         # End-to-end pseudotime inference wrapper
├── causal.py             # Causal inference and visualization interface
└── __init__.py           # Unified API access
```

## Components

### BaseWrapper (base.py)
- Abstract class for standardized `.fit()`, `.run()`, `.result` interface.
- Supports config saving via `.save_params()`.

### GraphWrapper (graph.py)
- Handles graph construction using `knn` or `radius` method.
- Allows extraction of subgraphs via strategies like `k_hop`, `bfs_python`, etc.
- Can convert graphs to NetworkX or PyTorch Geometric format.

### FeatureWrapper (feature.py)
- Microenvironment feature extraction interface.
- Supports built-in recipes, centre-type selection, graph and subgraph config overrides.
- Extracted features are stored in `AnnData.obsm`.

### PseudotimeWrapper (pseudotime.py)
- Unified interface for dimensionality reduction, clustering, and Slingshot pseudotime inference.
- Includes built-in `.plot()` method for embedding visualization.
- Provides easy access to embedding, clusters, pseudotime values, and calculated trend metrics.

### CausalWrapper (causal.py)
- One-stop interface for Granger causality (supports others via factory).
- Accepts a matrix from `.obsm`, with optional custom feature names.
- Handles pseudotime binning and effect estimation.
- Offers `.plot(kind="heatmap" | "bar" | "volcano")` for result visualization.

## Example Usage

```python
from tic.wrappers import GraphWrapper, FeatureWrapper, PseudotimeWrapper

# Build a graph
gw = GraphWrapper(method="knn", k=10)
adata = gw.fit(adata)

# Extract features
fw = FeatureWrapper(recipe="tme_default", centre_types=["Tumor"])
adata = fw.fit(adata)

# Run pseudotime
pw = PseudotimeWrapper(dr_method="umap", cluster_method="kmeans")
adata = pw.fit(adata)
pw.plot(kind="pseudotime")
```

## Notes

- Each wrapper is initialized with a config dataclass (e.g., `GraphConfig`, `FeatureConfig`, etc.).
- Wrappers ensure consistency, caching of results, and optional visualization capabilities.

---
Generated automatically for development documentation.
