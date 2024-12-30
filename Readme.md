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
# Space-Gm
## How to construct a graph:
you need : 
    cell_data for coordinates: CELL_ID,X,Y 
    cell_features : CELL_ID,SIZE
    cell_types: CELL_ID,CLUSTER_LABEL
    cell_expression: ACQUISITION_ID,CELL_ID,CD11b,CD14,CD15,CD163,CD20 ,,,
