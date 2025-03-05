"""
This module contains global constants used throughout the project. It includes:
  - Mappings for column names and file names.
  - Lists of biomarkers and cell types.
  - Definitions for general cell type groups.
  - Cutoff values and edge type definitions.
  - Mappings for cell representation methods and default representation pipelines.
"""

# List of all available biomarkers.
ALL_BIOMARKERS = [
    "CD11b", "CD14", "CD15", "CD163", "CD20", "CD21", "CD31", "CD34", 
    "CD3e", "CD4", "CD45", "CD45RA", "CD45RO", "CD68", "CD8", "CollagenIV", 
    "HLA-DR", "Ki67", "PanCK", "Podoplanin", "Vimentin", "aSMA"
]

# List of all cell types.
ALL_CELL_TYPES = [
    "APC", "B cell", "CD4 T cell", "CD8 T cell", "Granulocyte", "Lymph vessel",
    "Macrophage", "Naive immune cell", "Stromal / Fibroblast", "Tumor", 
    "Tumor (CD15+)", "Tumor (CD20+)", "Tumor (CD21+)", "Tumor (Ki67+)", 
    "Tumor (Podo+)", "Vessel", "Unassigned"
]

# Dictionary mapping general cell type groups to their corresponding subtypes.
GENERAL_CELL_TYPES = {
    "Immune": ["APC", "B cell", "CD4 T cell", "CD8 T cell", "Granulocyte", "Macrophage", "Naive immune cell"],
    "Tumor": ["Tumor", "Tumor (CD15+)", "Tumor (CD20+)", "Tumor (CD21+)", "Tumor (Ki67+)", "Tumor (Podo+)"],
    "Stromal": ["Stromal / Fibroblast"],
    "Vascular": ["Vessel", "Lymph vessel"],
    "Unassigned": ["Unassigned"],
}

# Mapping for file names using a format string with the region_id.
FILE_MAPPING = {
    'coords': "{region_id}.cell_data.csv",
    'features': "{region_id}.cell_features.csv",
    'types': "{region_id}.cell_types.csv",
    'expression': "{region_id}.expression.csv"
}

# Mapping for column names for each file type.
# The key is the file type and the value is a dictionary mapping the formal column name
# to the name used in the user's dataset.
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

# Mapping of cell representation methods to their corresponding function names.
CELL_REPRESENTATION_METHODS = {
    'biomarker_expression': 'get_center_cell_expression',
    'neighbor_cell_type_distribution': 'get_neighborhood_cell_type_distribution',
    'nn_embedding': 'get_nn_embedding'
}

# Mapping of representation method keys to their string identifiers.
REPRESENTATION_METHODS = {
    "raw_expression": "raw_expression",
    "neighbor_composition": "neighbor_composition",
    "nn_embedding": "nn_embedding"
}

# Default pipeline for cell representation.
# This list defines the order of representation methods to be applied.
DEFAULT_REPRESENTATION_PIPELINE = [
    REPRESENTATION_METHODS["raw_expression"],
    REPRESENTATION_METHODS["neighbor_composition"]
]

# GNN Training constant:
# Cutoff value for neighbor edge detection (55 pixels is approximately 20 µm).
NEIGHBOR_EDGE_CUTOFF = 55

# Definitions for different edge types.
EDGE_TYPES = {
    "neighbor": 0,
    "distant": 1,
    "self": 2,
}

# Cutoff for MicroE neighbor calculation (200 pixels is approximately 70 µm).
MICROE_NEIGHBOR_CUTOFF = 200