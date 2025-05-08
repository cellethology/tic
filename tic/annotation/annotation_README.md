# `tic.annotation`

The `tic.annotation` module provides a complete pipeline for **cell clustering** and **automatic cell-type annotation** in single-cell RNA-seq datasets. It integrates **Scanpy** for unsupervised clustering and **LLMs** for biological label inference based on marker gene profiles.

---

## 🧠 Module Structure

```bash
tic.annotation
├── annotation.py        # Core logic: clustering and top gene extraction
├── api.py               # High-level annotation interface
├── llm_caller.py        # LLM integration and JSON parsing
├── prompt.txt           # Prompt template for LLM annotation
├── llm_backends      # LLM backend handler
```

---

## 🔍 Key Functions & Classes

### 1. `assign_cell_clusters()`

Performs Scanpy-based clustering: filtering, normalization, PCA, KNN graph, Leiden clustering, and UMAP.

```python
from tic.annotation import assign_cell_clusters

adata = assign_cell_clusters(h5ad_path="your_data.h5ad", resolution=1.5)
```

This function annotates:
- `adata.obs["leiden"]`: cluster labels
- `adata.obsm["X_umap"]`: UMAP coordinates

---

### 2. `return_top_genes()`

Identifies top `n` marker genes per cluster using differential expression.

```python
top_genes = return_top_genes(adata, groupby="leiden", n_genes=5)
```

Returns:
```json
{
  "0": {"genes": [...], "scores": [...]},
  "1": {"genes": [...], "scores": [...]},
  ...
}
```

---

### 3. `LLMPredictor`

Wraps an LLM to generate **interpretable cell-type annotations** based on marker genes.

We default use openai as the LLM backend.So you need to set the api key for openai. Add OPENAI_API_KEY to your environment variables or pass the api key to the model_kwargs.
```python
from tic.annotation import LLMPredictor

llm = LLMPredictor(model_name="openai", model_kwargs={"api_key": "sk-..."})
annotation = llm.annotate_clusters(top_genes)
```

---

### 4. `annotate_adata()`

Main entrypoint: runs clustering → marker gene extraction → LLM cell-type annotation.

```python
from tic.annotation import annotate_adata

adata, annotation = annotate_adata(
    "example_data.h5ad",
    assign_params={"resolution": 1.0},
    llm_model_name="openai",
    llm_kwargs={"api_key": "..."},
    dataset_description="Mouse liver tissue",
    return_cluster_annotation=True
)
```

Adds `.obs["pred_cell_type"]` with LLM-derived labels.

---

## 📥 Input / 📤 Output

### Input:
- `.h5ad` or `AnnData` object with expression matrix

### Output:
- Cluster labels (`obs["leiden"]`)
- Top marker genes
- LLM-based annotation (`obs["pred_cell_type"]`)
- UMAP (`obsm["X_umap"]`)

---

## 🔧 Requirements

- Python ≥ 3.9
- `scanpy`, `anndata`
- OpenAI or compatible LLM API

---

## 📌 Example

```python
from tic.annotation import annotate_adata

adata, reasoning = annotate_adata(
    "data/sample.h5ad",
    llm_kwargs={"api_key": "sk-..."},
    return_cluster_annotation=True
)
```

---

## ✏️ Customization

- You may modify the prompt in `tic/annotation/prompt.txt` to suit your LLM and biological domain.
