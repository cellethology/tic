# tic/plotting/__init__.py
"""
tic.plotting: High-level plotting utilities for TIC.
"""

from .graph import plot_graph
from .pseudotime import scatter_embedding, plot_biomarker_trends
from .utils import moving_average, normalize
from .casual import plot_causal_heatmap, plot_causal_bar, plot_causal_volcano
from .metrics import plot_monotonicity_metrics_bar, plot_trend_metrics_bar
from .cross_region_heatmap import plot_cross_region_causal_heatmap
from .geometry import plot_cell_boundary, plot_fourier_normalized_shape, generate_fourier_reconstruction_gif
from .variability import plot_variability
from .gen_distribution import plot_gene_distributions_by_category
__all__ = [
    "plot_graph",
    "scatter_embedding",
    "plot_biomarker_trends",
    "moving_average",
    "normalize",
    "plot_causal_heatmap",
    "plot_causal_bar",
    "plot_causal_volcano",
    "plot_monotonicity_metrics_bar",
    "plot_trend_metrics_bar",
    "plot_cross_region_causal_heatmap",
    "plot_cell_boundary",
    "plot_fourier_normalized_shape",
    "generate_fourier_reconstruction_gif",
    "plot_variability",
    "plot_gene_distributions_by_category"
]