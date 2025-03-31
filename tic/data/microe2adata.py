"""
Module: tic.data.microe2adata

Provides a function to export MicroE center cells as rows in an AnnData object.
This includes inherent biomarker expression, extra representation features,
and a flattened neighbor biomarker matrix.
"""

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
    neighbor_cell_types: List[str] = ALL_CELL_TYPES,
) -> anndata.AnnData:
    """
    Export each MicroE's center cell as a row in an AnnData object.

    For each MicroE, this function:
      1. Computes extra representations using export_center_cell_with_representations.
      2. Constructs the inherent biomarker expression matrix (X) from the center cell.
      3. Extracts extra representations from additional features.
      4. Computes and flattens the neighbor biomarker matrix.
      5. Constructs obs metadata for the center cell.
      6. Uses build_ann_data to create a standardized AnnData object.

    Returns
    -------
    anndata.AnnData
        An AnnData object with:
         - X: center cell biomarker expression matrix.
         - obs: center cell metadata.
         - var: biomarker names.
         - obsm: extra representations and flattened neighbor biomarker matrix.
         - uns: additional information including neighbor biomarker feature names.
    """
    if not microe_list:
        return anndata.AnnData(
            X=np.empty((0, 0)), obs=pd.DataFrame(), var=pd.DataFrame(), obsm={}
        )

    cells = []
    obs_list = []
    expected_dims = {}
    rep_obsm = {}
    neighbor_rep_list = []
    X_rows = []

    for microe in microe_list:
        microe.export_center_cell_with_representations(
            representations=representations, model=model, device=device
        )
        center = microe.center_cell
        cells.append(center)

        # Build inherent biomarker expression row.
        bio_names = list(center.biomarkers.biomarkers.keys())
        bio_row = [center.biomarkers.biomarkers.get(bm, np.nan) for bm in bio_names] if bio_names else []
        X_rows.append(bio_row)

        # Build obs metadata.
        obs_dict = {
            "tissue_id": center.tissue_id,
            "cell_id": center.cell_id,
            "cell_type": center.cell_type,
            "size": center.size,
            "microe_neighbors_count": len(microe.neighbors),
        }
        obs_list.append(obs_dict)

        # Process extra representations.
        for rep in representations:
            vec = center.get_feature(rep)
            flat = np.ravel(vec) if vec is not None else np.array([])
            if rep not in expected_dims:
                expected_dims[rep] = flat.shape[0]
            target_dim = expected_dims[rep]
            if flat.shape[0] < target_dim:
                flat = np.pad(flat, (0, target_dim - flat.shape[0]), constant_values=0.0)
            elif flat.shape[0] > target_dim:
                flat = flat[:target_dim]
            rep_obsm.setdefault(rep, []).append(flat)

        # Compute flattened neighbor biomarker matrix.
        nb_matrix = microe.get_neighborhood_biomarker_matrix(
            biomarkers=neighbor_biomarkers, cell_types=neighbor_cell_types
        )
        neighbor_rep_list.append(nb_matrix.flatten())

    # Build X: pad each row to match the number of biomarker names from the first cell.
    biomarker_names = list(cells[0].biomarkers.biomarkers.keys()) if cells else []
    max_bio = len(biomarker_names)
    padded_X_rows = [
        row + [0.0] * (max_bio - len(row)) if len(row) < max_bio else row[:max_bio]
        for row in X_rows
    ]
    X_array = np.array(padded_X_rows, dtype=float)

    uns = {"data_level": "center cell"}
    if cells:
        uns["tissue_id"] = cells[0].tissue_id

    adata = build_ann_data(
        cells=cells, X=X_array, extra_obs=obs_list, uns=uns, feature_names=biomarker_names
    )

    # Add extra representation arrays to obsm.
    for rep, rep_list in rep_obsm.items():
        adata.obsm[rep] = np.vstack(rep_list)

    # Process flattened neighbor biomarker representation.
    flattened_length = len(neighbor_cell_types) * len(neighbor_biomarkers)
    neighbor_rep_list_padded = []
    for nb in neighbor_rep_list:
        if nb.shape[0] < flattened_length:
            nb = np.pad(nb, (0, flattened_length - nb.shape[0]), constant_values=0.0)
        elif nb.shape[0] > flattened_length:
            nb = nb[:flattened_length]
        neighbor_rep_list_padded.append(nb)
    adata.obsm["neighbor_biomarker"] = np.vstack(neighbor_rep_list_padded)

    # Build mapping from flattened index to "biomarker & cell type" names.
    neighbor_feature_names = [
        f"{bm}&{ct}" for ct in neighbor_cell_types for bm in neighbor_biomarkers
    ]
    adata.uns["neighbor_biomarker_feature_names"] = neighbor_feature_names

    print(f"Current AnnData Structure: {adata}")
    return adata