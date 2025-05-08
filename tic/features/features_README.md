# `tic.features`

The `tic.features` module provides a **pluggable and extensible framework** for extracting microenvironment features from spatial transcriptomics data. It supports both cell-level and neighborhood-level feature extraction via a registry-based system.

---

## 📦 Structure Overview

```bash
tic/features/
├── base.py              # Abstract base class for all extractors
├── registry.py          # Global registry and decorator for extractors
├── recipes.py           # Built-in recipe definitions
├── __init__.py          # High-level API: list, describe, extract, register
├── cell/
│   └── expression.py    # Centre cell gene expression
├── neighbourhood/
│   ├── composition.py
│   ├── celltype_gene_count.py
│   ├── gene_sum.py
│   ├── same_type_gene_sum.py
```

---

## 🧠 High-Level API

```python
from tic.features import extract, list_available, describe
```

### `extract(...)`
Main interface to extract features from a tissue-level AnnData object using predefined or custom recipes.

```python
out_adata = extract(
    adata,
    recipe="tme_default",
    centre_types=["Tumor Cells"],
    graph_params={"k": 15},
    subgraph_params={"strategy": "radius", "radius": 80}
)
```

Stores results in `.obsm[name]` for each extractor.

---

## 📚 Built-in Recipes

Available via `recipes.py` and `get_recipe()`:

- **cell_basic** – Centre cell only
- **tme_default** – Full tumor microenvironment:
  - `centre_gene`
  - `composition`
  - `neighbor_gene_sum`
  - `same_type_gene_sum`
  - `same_type_gene_average`
  - `celltype_gene_count`

---

## 🧩 Feature Extractors

### Cell-Level

- `CentreGene`: raw expression of the centre cell

### Neighbourhood-Level

- `NeighbourComposition`: fraction or count of cell types in the neighborhood
- `NeighbourGeneSum`: total expression in neighborhood
- `SameTypeGeneSum`: sum of expression from neighbors of the same type
- `SameTypeGeneAverage`: mean of expression from same-type neighbors
- `CelltypeGeneCount`: # of neighbors of each type expressing each gene

---

## 🔧 Custom Feature Extractors

You can define and register your own extractor:

```python
from tic.features.base import FeatureExtractor
from tic.features.registry import register

@register
class MyFeature(FeatureExtractor):
    name = "my_feature"

    def transform(self, adata, *, centre_idx, neighbour_idx):
        ...
        return vector
```

---

## 🧪 Development Notes

- Use `.obsp` to store graphs, `.obsm` to store extracted features.
- Use `FeatureExtractor.name` to define the key in `.obsm`.
- You can inspect registered extractors using `list_available()` or `describe()`.

---

## ✅ Example Usage

```python
from tic.features import extract

# Load an AnnData object (must have .obsm["spatial"])
adata = ...

# Extract full microenvironment features
out = extract(adata, recipe="tme_default", centre_types=["Tumor Cells"])
```

---

## 📎 Notes

- All extractors operate on neighborhoods defined by subgraph strategies (`radius`, `knn`, etc.).
- Modular design allows easy extension to multimodal or temporal features.
