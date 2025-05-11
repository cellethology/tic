# List of all available biomarkers.
UPMC_GENES = [
    "CD11b", "CD14", "CD15", "CD163", "CD20", "CD21", "CD31", "CD34",
    "CD3e", "CD4", "CD45", "CD45RA", "CD45RO", "CD68", "CD8", "CollagenIV",
    "HLA-DR", "Ki67", "PanCK", "Podoplanin", "Vimentin", "aSMA"
]

# UPMC_CELL_TYPES = [
#     "APC", "B cell", "CD4 T cell", "CD8 T cell", "Granulocyte", "Lymph vessel",
#     "Macrophage", "Naive immune cell", "Stromal / Fibroblast", "Tumor",
#     "Tumor (CD15+)", "Tumor (CD20+)", "Tumor (CD21+)", "Tumor (Ki67+)",
#     "Tumor (Podo+)", "Vessel", "Unassigned"
# ]
UPMC_CELL_TYPES = [
    "APC", "B cell", "CD4 T cell", "CD8 T cell", "Granulocyte", "Lymph vessel",
    "Macrophage", "Naive immune cell", "Stromal / Fibroblast", "Tumor",
    "Tumor (CD15+)", "Tumor (CD20+)", "Tumor (CD21+)", "Tumor (Ki67+)",
    "Tumor (Podo+)", "Vessel"
]

# Dictionary mapping general cell type groups to their corresponding subtypes.
GENERAL_CELL_TYPES = {
    "Immune": [
        "APC", "B cell", "CD4 T cell", "CD8 T cell", "Granulocyte", "Macrophage",
        "Naive immune cell"
    ],
    "Tumor": [
        "Tumor", "Tumor (CD15+)", "Tumor (CD20+)", "Tumor (CD21+)",
        "Tumor (Ki67+)", "Tumor (Podo+)"
    ],
    "Stromal": ["Stromal / Fibroblast"],
    "Vascular": ["Vessel", "Lymph vessel"],
    "Unassigned": ["Unassigned"],
}

UPMC_EPITHELIAL_GENES = ["PanCK"]
UPMC_MESENCHYMAL_GENES = ["Vimentin", "aSMA", "CollagenIV"]
UPMC_EMT_GENES = ["CollagenIV", "PanCK", "Vimentin", "aSMA"]


def concat_fea_adata(fea_adata_list):
    '''
    concat fea_adata_list into one adata
    '''
    pass


