# TIC: Temporal Inference of Cells

**Version**: 2.0.0  
**Description**: TIC is a modular Python package for analyzing dynamic cellular microenvironments using spatial transcriptomic data. It supports end-to-end pipelines for:
- LLM-based cell type annotation
- Microenvironment feature extraction
- Pseudotime inference
- Causal inference

---

## 📦 Module Structure & Public API

### 📁 Data I/O
- `loader`: Loaders for different datasets: load_codex_upmc(will automatically download), load_xenium_pancreas_cancer(will automatically download), load_xenium_colorectal_cancer(can load but not automatically download)

### 🧬 Annotation (LLM-assisted)
- `assign_cell_clusters`: Preprocessing pipeline for clustering and marker gene extraction.
- `return_top_genes`: Marker gene ranking.
- `LLMPredictor`: LLM-based cell type annotation engine.
- `get_llm`: Load OpenAI LLM interface.

### 🔄 Pipeline
- `pipeline.pseudotime`: Access to full pseudotime inference pipeline (via `PseudotimePipeline`).

### 📐 Feature Extraction
- `features.base`: Feature extractors for cellular neighborhoods.
- `features.registry`: Registry of built-in microenvironment recipes.
- `features.recipes`: Built-in recipes for feature extraction.
- `FeatureWrapper`: Wrapper for feature extraction.

### 📊 Pseudotime Inference
- `pipeline.pseudotime`: Access to full pseudotime inference pipeline (via `PseudotimePipeline`).
- `PseudotimeWrapper`: Wrapper for pseudotime inference.(will return adata with .obs['pseudotime'], directly run `pw.fit(adata)` to get the result,this will run the whole pipeline)

### 📊 Causal Inference
- `causal.causal_input`: Causal input class for causal inference.
- `causal.base`: Base class for causal methods.
- `causal.factory`: Factory to dynamically instantiate causal inference methods.
- `CausalWrapper`: Wrapper for causal inference.

### 📈 Monotonicity & Trend Metrics
- `calculate_monotonicity`: Spearman/Kendall correlation.
- `calculate_trend`: Mann-Kendall or linear regression trend.
- `rank_by_monotonicity`: Rank multiple series by monotonicity.
- `rank_by_trend`: Rank multiple series by trend statistics.

---

## 🧪 Example Usage

```python
from tic.wrappers import FeatureWrapper, PseudotimeWrapper
import scanpy as sc

adata = sc.read_h5ad("example.h5ad")

# 1. Feature extraction
fw = FeatureWrapper(recipe='tme_default', centre_types=['Tumor'])
adata = fw.fit(adata)

# 2. Pseudotime inference
pw = PseudotimeWrapper(rep_key='centre_gene', n_clusters=3, dr_method='pca')
adata = pw.fit(adata)
```

---


## 📬 Contact
Developed by Zhang Jiahao et al., Westlake University. For feedback or bugs, please open a GitHub issue or contact the maintainers.