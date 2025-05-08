# `tic.pseudotime`

The `tic.pseudotime` module provides a modular pipeline for **pseudotime inference** in spatial or single-cell datasets. It includes wrappers for **dimensionality reduction**, **clustering**, and pseudotime algorithms like **Slingshot**, with a clear design to support additional methods.

---

## 📦 Module Structure

```bash
tic/pseudotime/
├── pp/         # Preprocessing (dimensionality reduction, clustering)
│   ├── clustering.py
│   └── dimensionality.py
├── tl/         # Pseudotime inference tools
│   ├── base.py
│   └── slingshot.py
```

---

## 🧬 Preprocessing

### `DimensionalityReduction`

Unified wrapper for PCA, t-SNE, UMAP, etc.

```python
from tic.pseudotime.pp import DimensionalityReduction

dr = DimensionalityReduction(method="umap", n_components=2)
X_reduced = dr.fit_transform(expression_matrix)
```

Supported methods:
- `"pca"`, `"kernel_pca"`, `"tsne"`, `"umap"`, `"mds"`, `"isomap"`, `"lle"`

---

### `Clustering`

Unified interface for common clustering algorithms.

```python
from tic.pseudotime.pp import Clustering

clusterer = Clustering(method="kmeans", n_clusters=5)
labels = clusterer.fit_predict(X_reduced)
```

Supported methods:
- `"kmeans"`, `"agg"`, `"dbscan"`, `"mean_shift"`, `"spectral"`, `"birch"`, `"gmm"`

---

## 🧪 Pseudotime Inference

### `SlingshotMethod`

Wrapper around [`pyslingshot`](https://github.com/fairinternal/pyslingshot), integrated into TIC’s modular API.

```python
from tic.pseudotime.tl import SlingshotMethod

sl = SlingshotMethod(start_node=0)
pseudotime = sl.fit_predict(X_reduced, labels)
```

Adds diagnostics if `output_dir` is specified (e.g., trajectory plots).

---

## 🧩 Interface & Extensibility

### `PseudotimeMethod` (Abstract Base Class)

All pseudotime algorithms must subclass `PseudotimeMethod` and implement:

```python
def fit_predict(self, embeddings, labels, *, output_dir=None) -> np.ndarray
```

This allows seamless plug-in of custom methods.

---

## ✅ Example Workflow

```python
from tic.pseudotime.pp import DimensionalityReduction, Clustering
from tic.pseudotime.tl import SlingshotMethod

# 1. Reduce dimension
dr = DimensionalityReduction(method="umap")
X_emb = dr.fit_transform(expression_matrix)

# 2. Cluster
cl = Clustering(method="kmeans", n_clusters=6)
labels = cl.fit_predict(X_emb)

# 3. Infer pseudotime
sl = SlingshotMethod(start_node=0)
pseudotime = sl.fit_predict(X_emb, labels)
```

---
