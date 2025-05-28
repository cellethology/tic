# tic.data

Low- and high-level utilities for downloading, extracting, validating and loading  
spatial transcriptomics datasets into `anndata.AnnData`.

## 📦 Package Structure

```bash
tic/data
├── io.py                 # download_file, extract_zip, cache management
├── utils.py              # AnnData checks & biomarker/cell-type helpers
├── xenium
│   ├── download.py       # Xenium dataset registry & download wrappers
│   └── loader.py         # load_xenium_dataset → AnnData
├── codex
│   ├── download.py       # CODEX dataset registry & download wrappers
│   └── loader.py         # list_regions, load_region → AnnData
└── __init__.py           # exposes all public APIs
```

---

## 🔧 Low-level I/O

- **`download_file(url, dest)`**  
  Fetch a URL into a local file (skips if exists).

- **`extract_zip(archive, out_dir, cleanup=False, members=None)`**  
  Unpack a ZIP archive, optionally removing it afterwards.

- **`remove_cache(cache_dir)`**  
  Delete entire cache directory tree.

- **`list_datasets(cache_dir)`** → `List[str]`  
  All top-level dataset names in cache.

- **`remove_dataset(name, cache_dir)`**  
  Delete a single dataset directory.

---

## 🔍 Validation & Metadata

- **`check_spatial_anndata(adata)`**  
  Ensure `.X`, `.var_names`, and `.obsm["spatial"]` exist; densify sparse matrices.

- **`check_EMT_genes(adata)`**  
  Confirm EMT marker genes are in `adata.var_names`.

- **`get_cell_types(adata)`** → `List[str]`  
  Ordered list of cell types from `.obs["cell_type"]` or fallback to `["Unassigned"]`.

- **`get_biomarkers(adata)`** → `List[str]`  
  Ordered biomarkers from `adata.var_names` or auto-generated names.

---

## 🧬 Xenium Datasets

### `ensure_xenium_dataset(name, cache_dir, force=False)` → `Path`

Registry keys:

- `xenium_ffpe_human_breast`
- `xenium_pancreas_cancer`

Downloads and extracts the 10x Xenium `.zip` into `cache_dir/name`.

### `download_xenium_dataset(name, cache_dir, force=False)`

Wrapper around `ensure_xenium_dataset`, discarding the return path.

### `load_xenium_dataset(name, cache_dir, force_download=False,  
                         normalize=None, n_cells=None, include_mask=False)`  
→ `AnnData`

Loads a Xenium dataset as an `AnnData` with:

- `.X`: expression matrix  
- `.obs`: cell metadata (including centroids in `x_centroid`, `y_centroid`)  
- `.obsm["spatial"]`: centroids array  
- `.uns["tissue_id"]`, `.uns["data_level"]`  
- optional polygon masks in `.uns["cell_boundaries"]`.

```python
from tic.data import load_xenium_dataset
adata = load_xenium_dataset(
    "xenium_pancreas_cancer",
    normalize="counts",
    n_cells=5000,
    include_mask=True,
)
```

---

## 🧪 CODEX Datasets

### `ensure_codex_dataset(dataset, cache_dir)` → `Path`

Registry keys:

- `upmc`, `charville`, `dfci`

Fetches and flattens the CODEX `.zip` from Zenodo into `cache_dir/codex_<dataset>`.

### `download_codex_dataset(dataset, cache_dir)`

Convenience wrapper for `ensure_codex_dataset`.

### `list_regions(dataset, cache_dir)` → `List[str]`

Enumerate available region IDs (e.g. `UPMC_c001_v001_r001_reg001`).

### `load_region(dataset, region_id, cache_dir)` → `AnnData`

Loads one CODEX region into an `AnnData`:

- `.X`: marker expression (`float32`)  
- `.obs`: cell metadata (`cell_id`, `cell_type`, `size`)  
- `.obsm["spatial"]`: 2D coordinates  
- `.uns["tissue_id"]`, `.uns["data_level"]`

```python
from tic.data import list_regions, load_region
regions = list_regions("upmc")
adata = load_region("upmc", regions[0])
```

---

_All datasets are cached under `~/.cache/tic/` by default._
