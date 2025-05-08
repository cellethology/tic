
# `tic.plotting`: Visualization Utilities for Temporal Inference of Cells

The `tic.plotting` module provides high-level plotting utilities tailored to the TIC pipeline for spatial, pseudotime, trend, and causal inference visualization. These tools are designed to integrate seamlessly with `AnnData` objects.

---

## 📦 Submodules & Key Functions

### 1. `graph.py`
- `plot_graph(...)`:
  - Visualize the 2D spatial layout of cells and their adjacency graph.
  - Optional coloring by cell type.
- `plot_spot(...)`, `plot_spot_inner_cells(...)`:
  - Visualize cells within a spatial domain (spot) and optionally color by cell type.

### 2. `pseudotime.py`
- `scatter_embedding(...)`:
  - UMAP/t-SNE scatterplot, colored by pseudotime or cluster label.
- `plot_biomarker_trends(...)`:
  - Line plots showing biomarker expression trends along pseudotime.
  - Supports raw/bin/bin+normalize on x-axis; normalize/smooth on y-axis.

### 3. `metrics.py`
- `plot_monotonicity_metrics_bar(...)`:
  - Bar plot highlighting biomarkers ranked by monotonicity score.
  - EMT genes are color-coded by type (Epithelial, Mesenchymal, EMT TF).
- `plot_trend_metrics_bar(...)`:
  - Similar to above but based on trend statistics (e.g., Mann-Kendall Z).

### 4. `casual.py` (Causal Analysis Visualizations)
- `plot_causal_heatmap(...)`:
  - Heatmap of predictor-biomarker pairs colored by effect size or -log10(p).
- `plot_causal_bar(...)`:
  - Bar plot of top predictors sorted by p-value.
- `plot_causal_volcano(...)`:
  - Volcano plot with log2(effect) vs -log10(p) highlighting significant predictors.

### 5. `utils.py`
- `moving_average(...)`, `normalize(...)`, `default_y_transform(...)`, `fill_nan_with_interp(...)`:
  - Utility functions for 1D data smoothing, normalization, and NaN handling.

---

## 🧪 Dependencies
- `matplotlib`
- `seaborn`
- `anndata`
- `numpy`
- `pandas`

---

## 📌 Notes
- All functions accept optional `save_path` for automated figure export.
- Gene-specific color palettes are hardcoded for consistency across plots (EMT, epithelial, mesenchymal, etc.).
- Plots are designed to accommodate `AnnData` structure used in TIC, especially `.uns['causal_results']`, `.obs['pseudotime']`, and `.obsm['spatial']`.

