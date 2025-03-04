COLUMN_MAPPING = {
    'region_id': 'REGION_ID', # pseudotime.csv
    'cell_id': 'CELL_ID', # pseudotime.csv
    'pseudotime': 'PSEUDOTIME', # pseudotime.csv
    'ACQUISITION_ID': 'REGION_ID', # {expression}.csv
    # 'CELL_ID': 'CELL_ID', # {expression}.csv
    'CLUSTER_LABEL': 'CELL_TYPE' # {cell_types}.csv
}
ALL_BIOMARKERS = ["CD11b", "CD14", "CD15", "CD163", "CD20", "CD21", "CD31", "CD34", "CD3e", "CD4", "CD45", "CD45RA", "CD45RO", "CD68", "CD8", "CollagenIV", "HLA-DR", "Ki67", "PanCK", "Podoplanin", "Vimentin", "aSMA"]
ALL_CELL_TYPES = ["APC", "B cell", "CD4 T cell", "CD8 T cell", "Granulocyte", "Lymph vessel", "Macrophage", "Naive immune cell", "Stromal / Fibroblast", "Tumor", "Tumor (CD15+)", "Tumor (CD20+)", "Tumor (CD21+)", "Tumor (Ki67+)", "Tumor (Podo+)", "Vessel", "Unassigned"]


NEIGHBOR_EDGE_CUTOFF = 55 # 55pixels ~ 20 um

EDGE_TYPES = {
    "neighbor": 0,
    "distant": 1,
    "self": 2,
}

MICROE_NEIGHBOR_CUTOFF = 200 # 200pixels ~ 70 um