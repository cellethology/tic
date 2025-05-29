# Epthelial Markers
# EPITHELIAL_GENES = {'CDH1','EPCAM','DSP','KRT5', 'KRT8','KRT18','KRT19','LAMA1','LAMB1','LAMC1'}
import os

EPITHELIAL_GENES = {'CDH1','EPCAM','DSP'}
# Mesenchymal Markers
MESENCHYMAL_GENES = {'CDH2','VIM','ACTA2','FN1'}

EMT_TF = {'ZEB1','ZEB2','SNAI1','SNAI2','TWIST1'}

# EMT Markers
EMT_GENES = EPITHELIAL_GENES.union(MESENCHYMAL_GENES).union(EMT_TF)

# Default key in adata:
DEFAULT_KEY = {
    'cell_type': 'cell_type', # the key in adata.obs to store the cell type, see: tic.data.utils.get_cell_types
    'cell_types': 'cell_types', # the key in adata.obsm to store the cell types, see: tic.data.utils.get_cell_types
    'biomarkers': 'biomarkers', # the key in adata.uns to store the biomarkers, see: tic.data.utils.get_biomarkers
    'connectivities_key': 'connectivities', # the key in adata.obsp to store the connectivities, see: tic.graph.utils.get_connectivities_key
    'graph_params': 'graph_params', # the key in adata.uns to store the graph parameters, see: tic.graph.pp.neighbors.compute_neighbors
    'rp_reduced': 'rp_reduced', # the key in adata.obsm to store the reduced embedding, see: tic.wrappers.pseudotime.PseudotimeWrapper
    'cluster': 'cluster', # the key in adata.obs to store the cluster, see: tic.wrappers.pseudotime.PseudotimeWrapper
    'pseudotime': 'pseudotime', # the key in adata.obs to store the pseudotime, see: tic.wrappers.pseudotime.PseudotimeWrapper
    'causal_results': 'causal_results', # the key in adata.uns to store the causal results, see: tic.wrappers.causal.CausalWrapper
    'pipeline_config': 'pipeline_config', # the key in adata.uns to store the pipeline config, see: tic.pipeline.pseudotime.PseudotimePipeline
}

# Default cache directory for tic will be stored here: ~/.cache/tic if not set in environment variable
DEFAULT_DATACACHE_DIR = os.environ.get(
    "DEFAULT_DATACACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "tic")
)
