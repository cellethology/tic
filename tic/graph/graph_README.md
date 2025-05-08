# `tic.graph`

The `tic.graph` module provides a **Scanpy-inspired graph processing interface** tailored for spatial transcriptomics data. It supports preprocessing (graph construction), downstream analysis (subgraph extraction, metrics), and export to third-party libraries (e.g., PyG, NetworkX).

---

## 📦 Structure Overview

```bash
tic/graph/
├── pp/              # Preprocessing: graph construction & spot assignment
├── tl/              # Tools: subgraph extraction & graph-based metrics
├── io.py            # Export to PyG, NetworkX
├── utils.py         # Shared utilities
├── _typing.py       # Shared type aliases
```

---

## ⚙️ Core Functionalities

### 1. Build Spatial Graphs (`pp.compute_neighbors()`)

```python
from tic.graph.pp import compute_neighbors
compute_neighbors(adata, method="knn", k=10)
```

📌 Methods:
- `"knn"`: k-nearest neighbors
- `"radius"`: fixed distance threshold
- `"voronoi"`: Voronoi-based connectivity

Result stored in `.obsp["connectivities"]`

---

### 2. Assign Spatial Spots

```python
from tic.graph.pp import assign_spots, aggregate_by_spot
assign_spots(adata, method="grid", grid_x=10, grid_y=10)
spot_adata = aggregate_by_spot(adata)
```

---

### 3. Subgraph Extraction (`tl.extract_subgraph()`)

```python
from tic.graph.tl import extract_subgraph

# kNN subgraph
nodes = extract_subgraph(adata, center=0, strategy="knn", k=6)

# radius-based subgraph
nodes = extract_subgraph(adata, center=0, strategy="radius", radius=100)

# NetworkX graph
G = extract_subgraph(adata, center=0, strategy="radius", return_type="networkx")
```

📌 Supported strategies:
- `"knn"` / `"radius"`
- `"k_hop_shortest"` / `"k_hop_pyg"`
- `"bfs_python"` / `"slice_adj"`

---

### 4. Graph Metrics (`tl.local_degree()`)

```python
from tic.graph.tl import local_degree
deg = local_degree(adata, center=0)
```

---

### 5. Export to PyG / NetworkX

```python
from tic.graph.io import to_networkx, to_pyg
G = to_networkx(adata)
pyg_data = to_pyg(adata)
```

---

## 🧪 Example

```python
from tic.graph.pp import compute_neighbors
from tic.graph.tl import extract_subgraph

compute_neighbors(adata, method="knn", k=8)
sub_nodes = extract_subgraph(adata, center=0, strategy="k_hop_shortest", hop=2)
```

---

## 🧠 Notes

- `.obsp["connectivities"]` stores adjacency matrix
- `.uns["graph_params"]` stores graph construction metadata
- Flexible export and conversion to third-party graph libraries
