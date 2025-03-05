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
GENERAL_CELL_TYPES = {
    "Immune": ["APC", "B cell", "CD4 T cell", "CD8 T cell", "Granulocyte", "Macrophage", "Naive immune cell"],
    "Tumor": ["Tumor", "Tumor (CD15+)", "Tumor (CD20+)", "Tumor (CD21+)", "Tumor (Ki67+)", "Tumor (Podo+)"],
    "Stromal": ["Stromal / Fibroblast"],
    "Vascular": ["Vessel", "Lymph vessel"],
    "Unassigned": ["Unassigned"],
}


NEIGHBOR_EDGE_CUTOFF = 55 # 55pixels ~ 20 um

EDGE_TYPES = {
    "neighbor": 0,
    "distant": 1,
    "self": 2,
}

MICROE_NEIGHBOR_CUTOFF = 200 # 200pixels ~ 70 um


# Here is an example of how to use the COLUMN_MAPPING and FILE_MAPPING constants:
FILE_MAPPING = {
    'coords': "{region_id}.cell_data.csv",
    'features': "{region_id}.cell_features.csv",
    'types': "{region_id}.cell_types.csv",
    'expression': "{region_id}.expression.csv"
}

# The key is the formal name of the column, and the value is the name used in the your own dataset.
# Do column mapping for each file type.
COLUMN_MAPPING = {
    'coords': {
        'CELL_ID': 'ID',
        'X': 'X_COORD',
        'Y': 'Y_COORD'
    },
    'features': {
        'CELL_ID': 'ID',
        'SIZE': 'CELL_SIZE'
    },
    'types': {
        'CELL_ID': 'ID',
        'CELL_TYPE': 'TYPE'
    },
    'expression': {
        'CELL_ID': 'ID'
    }
}

CELL_REPRESENTATION_METHODS = {
    'biomarker_expression': 'get_center_cell_expression',
    'neighbor_cell_type_distribution': 'get_neighborhood_cell_type_distribution',
    'nn_embedding': 'get_nn_embedding'
}

REPRESENTATION_METHODS = {
    "raw_expression": "raw_expression",
    "neighbor_composition": "neighbor_composition",
    "nn_embedding": "nn_embedding"
}

DEFAULT_REPRESENTATION_PIPELINE = [
    REPRESENTATION_METHODS["raw_expression"],
    REPRESENTATION_METHODS["neighbor_composition"]
]