# `tic.data`

The `tic.data` module provides a comprehensive interface for **downloading**, **loading**, and **validating** spatial transcriptomics datasets including **Codex UPMC**, **Xenium Pancreas Cancer**, and **Xenium Colorectal Cancer** (WIP). It also includes utility functions for data I/O and biomarker metadata management.

---

## 📦 Module Structure

```bash
tic.data
├── io.py                 # Download and unzip dataset files
├── loader.py             # Load datasets into AnnData format
├── utils.py          # Input checks and biomarker utilities
├── __init__.py
```

---

## 📥 Dataset Download

### `download_codex_dataset()`
Downloads and unpacks the Codex UPMC dataset from Zenodo.

### `download_xenium_pancreas_cancer_data()`
Downloads Xenium human pancreas cancer data from 10x Genomics.

### `download_xenium_colorectal_cancer_data()`
Stub for future Xenium colorectal data integration.(Not available yet)

---

## 🧬 Dataset Loaders

### `load_codex_dataset()`
We currently support three Codex datasets:
- Codex-UPMC
- Codex-Charville
- Codex-DFCI

Loads Codex dataset as an `AnnData` object with:
- `.X`: marker expressions
- `.obs`: cell_id, cell_type, size
- `.obsm["spatial"]`: 2D coordinates
- `.uns`: tissue_id, data_level

### `load_xenium_pancreas_cancer()`

Loads Xenium dataset with optional filtering:
```python
adata = load_xenium_pancreas_cancer(normalize_method="counts", n_cells=5000)
```

- Auto-detects circular spatial region
- Supports area-based or count-based normalization

### `load_xenium_colorectal_cancer()`

Prepares `.X`, `.var_names`, and adds `spot` type metadata (requires `.h5ad` file).

---

## 🔍 Utils

### `check_spatial_anndata()`

Checks and validates AnnData for spatial transcriptomics compatibility:
- Presence of `.X`, `.var_names`, `.obsm["spatial"]`
- Converts sparse matrix to dense if needed

### `check_EMT_genes()`

Validates whether EMT marker genes are available in `.var_names`.

### `get_cell_types()`

Returns ordered list of cell types from `.obs["cell_type"]` or fallback.

### `get_biomarkers()`

Returns list of biomarker names from `.var_names`.

---

## ✅ Example Usage

```python
from tic.data import load_codex_upmc, check_spatial_anndata

adata = load_codex_upmc(region_id="UPMC_c001_v001_r001_reg001")
adata = check_spatial_anndata(adata)
```

---

## 📎 Notes

- All datasets are cached under `~/.cache/tic/`.
- `load_*` functions return `AnnData` objects ready for analysis.
