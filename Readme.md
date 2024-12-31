# TIC
Temporal Inference of Cells 

To use TIC, you need to install these modules:
1. space-gm: to construct graph for cell micro-environment
2. slingshot: this is the basic algrithm for TIC to do pseudo-time analysis
```Python
git submodule add https://github.com/kstreet13/slingshot.git tools/slingshot
git submodule add https://gitlab.com/enable-medicine-public/space-gm.git tools/space-gm

cd tools/space-gm
pip install -e.

cd ../..
pip install -e.
```
## Project Framework
### PseudoTime Analysis output dir
```
output_dir/
├── embedding_analysis/
│   ├── {embedding_key}/
│   │   ├── expression/
│   │   │   ├── biomarker_trends_vs_pseudotime.png
│   │   │   ├── aggregated_expression_data.csv
│   │   │   └── transform_logs.txt
│   │   ├── pseudotime/
│   │   │   ├── start_node_{start_node}/
│   │   │   │   ├── pseudotime.csv
│   │   │   │   ├── biomarker_trends_vs_pseudotime.png
│   │   │   │   ├── neighborhood_composition_vs_pseudotime.png
│   │   │   │   └── aggregated_composition_data.csv
│   │   │   └── pseudotime_overview.png
│   │   ├── clustering/
│   │   │   ├── umap_embeddings_vs_cell_types.png
│   │   │   └── cluster_summary.csv

```
# Space-Gm
## How to construct a graph:
you need : 
    cell_data for coordinates: CELL_ID,X,Y 
    cell_features : CELL_ID,SIZE
    cell_types: CELL_ID,CLUSTER_LABEL
    cell_expression: ACQUISITION_ID,CELL_ID,CD11b,CD14,CD15,CD163,CD20 ,,,
