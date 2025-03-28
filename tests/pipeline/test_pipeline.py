# tests/pipeline/test_pipeline.py

from tic.data.graph_feature import edge_attr_fn, edge_index_fn, node_feature_fn
from tic.data.microe2adata import export_center_cells
from tic.data.tissue import Tissue
from tic.pipeline.causal import run_causal_pipeline
from tic.pipeline.pseudotime import run_pseudotime_pipeline
from utils.dataload import process_region_to_anndata


example_data_dir = "/Users/zhangjiahao/Project/tic/data/example/Raw"
example_region = "UPMC_c001_v001_r001_reg001"

example_adata = process_region_to_anndata(raw_dir=example_data_dir,region_id=example_region)

example_tissue = Tissue.from_anndata(adata=example_adata, tissue_id=example_region)

example_tissue.to_graph(node_feature_fn=node_feature_fn,edge_attr_fn=edge_attr_fn, edge_index_fn=edge_index_fn)

# extract microe
example_micro_list = [example_tissue.get_microenvironment(center_cell_id=i) for i in list(example_tissue.cell_dict.keys())]

center_cells = export_center_cells(microe_list=example_micro_list, representations=["raw_expression", "neighbor_composition"])

cells_with_time = run_pseudotime_pipeline(adata=center_cells, copy=True)

infered_result = run_causal_pipeline(adata=cells_with_time, y_biomarker="PanCK", x_variable= None ,copy=True).uns['causal_results']

print(infered_result)

