# tic/data/microe2adata.py

from typing import List
import numpy as np
import pandas as pd
import anndata
from tic.data.microe import MicroE
from tic.data.utils import build_ann_data
from tic.constant import ALL_BIOMARKERS, ALL_CELL_TYPES

def export_center_cells(
    microe_list: List[MicroE],
    representations: List[str] = ["raw_expression", "neighbor_composition", "nn_embedding"],
    model=None,
    device=None,
    neighbor_biomarkers: List[str] = ALL_BIOMARKERS,
    neighbor_cell_types: List[str] = ALL_CELL_TYPES
) -> anndata.AnnData:
    """
    Export each MicroE's center cell as a row in an AnnData object, where:
      - X is the inherent biomarker expression matrix from the center cell.
      - obs contains cell metadata (tissue_id, cell_id, cell_type, size, microe_neighbors_count)
      - obsm contains additional representation features (e.g., "raw_expression", "neighbor_composition", "nn_embedding")
        and also a flattened neighbor biomarker matrix.
      
    For each MicroE, the function:
      1. Computes extra representations by calling export_center_cell_with_representations.
      2. Uses the center cell's inherent biomarker expression to build X (as in cell.to_anndata).
      3. Extracts requested representations from the center cell's additional_features.
      4. Computes the neighbor biomarker matrix via get_neighborhood_biomarker_matrix, flattens it,
         and stores it in obsm under the key "neighbor_biomarker".
         It also creates a feature mapping (e.g., "PanCK&Tumor", "CD3&Tumor", etc.) and stores it in uns.
      5. Constructs obs metadata for each center cell.
      6. Uses build_ann_data to create a standardized AnnData object.
    
    Returns:
      An AnnData object with:
         - X: biomarker expression matrix from the center cell.
         - obs: metadata for each center cell.
         - var: biomarker names.
         - obsm: contains "spatial" (cell positions), extra representations, and "neighbor_biomarker"
                 which is a flattened neighbor biomarker matrix.
         - uns: includes "data_level" and "neighbor_biomarker_feature_names".
    """
    if not microe_list:
        return anndata.AnnData(X=np.empty((0, 0)), obs=pd.DataFrame(), var=pd.DataFrame(), obsm={})

    cells = []
    obs_list = []
    # For each representation, record the expected dimension from the first center cell providing it.
    expected_dims = {}
    # For storing extra representation features from the center cell in obsm.
    rep_obsm = {}
    # For storing the flattened neighbor biomarker data.
    neighbor_rep_list = []

    # Build X from center cell biomarkers.
    X_rows = []
    for microe in microe_list:
        # 1. Compute extra representations.
        microe.export_center_cell_with_representations(
            representations=representations,
            model=model,
            device=device
        )
        center = microe.center_cell
        cells.append(center)
        # 2. Use center cell biomarkers for X.
        bio_names = list(center.biomarkers.biomarkers.keys())
        if bio_names:
            bio_row = [center.biomarkers.biomarkers.get(bm, np.nan) for bm in bio_names]
        else:
            bio_row = []
        X_rows.append(bio_row)
        # 3. Build obs metadata.
        obs_dict = {
            "tissue_id": center.tissue_id,
            "cell_id": center.cell_id,
            "cell_type": center.cell_type,
            "size": center.size,

            # additional feature
            "microe_neighbors_count": len(microe.neighbors)
        }
        obs_list.append(obs_dict)
        # 4. Process each requested representation.
        for rep in representations:
            vec = center.get_feature(rep)
            if vec is not None:
                flat = np.ravel(vec)
            else:
                flat = np.array([])
            if rep not in expected_dims:
                expected_dims[rep] = flat.shape[0]
            target_dim = expected_dims[rep]
            if flat.shape[0] < target_dim:
                flat = np.pad(flat, (0, target_dim - flat.shape[0]), constant_values=0.0)
            elif flat.shape[0] > target_dim:
                flat = flat[:target_dim]
            if rep not in rep_obsm:
                rep_obsm[rep] = []
            rep_obsm[rep].append(flat)
        # 5. Compute neighbor biomarker matrix (excluding center cell).
        # This matrix is of shape (n_cell_types, n_biomarkers)
        nb_matrix = microe.get_neighborhood_biomarker_matrix(
            biomarkers=neighbor_biomarkers, cell_types=neighbor_cell_types
        )
        # Flatten the matrix into a 1D array.
        nb_flat = nb_matrix.flatten()
        neighbor_rep_list.append(nb_flat)

    # Build X from inherent biomarkers.
    if cells:
        # Use biomarker names from the first cell.
        biomarker_names = list(cells[0].biomarkers.biomarkers.keys())
    else:
        biomarker_names = []
    max_bio = len(biomarker_names)
    padded_X_rows = []
    for row in X_rows:
        if len(row) < max_bio:
            row = row + [0.0] * (max_bio - len(row))
        elif len(row) > max_bio:
            row = row[:max_bio]
        padded_X_rows.append(row)
    X_array = np.array(padded_X_rows, dtype=float)
    
    uns = {"data_level": "center cell"}
    if cells:
        uns["tissue_id"] = cells[0].tissue_id

    # Build AnnData using the helper function.
    adata = build_ann_data(
        cells=cells,
        X=X_array,
        extra_obs=obs_list,
        uns=uns,
        feature_names=biomarker_names
    )
    
    # Add the extra representation arrays to obsm.
    for rep, rep_list in rep_obsm.items():
        rep_array = np.vstack(rep_list)
        adata.obsm[rep] = rep_array

    # Now add the flattened neighbor biomarker representation.
    # All microe instances use the same cell_types and biomarkers, so the flattened length is:
    # flattened_length = len(neighbor_cell_types) * len(neighbor_biomarkers)
    flattened_length = len(neighbor_cell_types) * len(neighbor_biomarkers)
    # Ensure each entry in neighbor_rep_list has this length.
    neighbor_rep_list_padded = []
    for nb in neighbor_rep_list:
        if nb.shape[0] < flattened_length:
            nb = np.pad(nb, (0, flattened_length - nb.shape[0]), constant_values=0.0)
        elif nb.shape[0] > flattened_length:
            nb = nb[:flattened_length]
        neighbor_rep_list_padded.append(nb)
    neighbor_array = np.vstack(neighbor_rep_list_padded)
    adata.obsm["neighbor_biomarker"] = neighbor_array

    # Also record the mapping from flattened index to (cell type & biomarker)
    neighbor_feature_names = []
    for ct in neighbor_cell_types:
        for bm in neighbor_biomarkers:
            neighbor_feature_names.append(f"{bm}&{ct}")
    adata.uns["neighbor_biomarker_feature_names"] = neighbor_feature_names

    print(f"Current AnnData Structure: {adata}")
    return adata