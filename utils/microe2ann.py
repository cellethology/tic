from typing import List, Optional
import numpy as np
import pandas as pd
import anndata
from tic.data.microe import MicroE
from tic.data.utils import build_ann_data


def export_center_cells(
    microe_list: List[MicroE],
    representations: List[str] = ["raw_expression", "neighbor_composition", "nn_embedding"],
    model=None,
    device=None
) -> anndata.AnnData:
    """
    Export each MicroE's center cell as a row in an AnnData object, where:
      - X is the biomarker expression matrix (as in cell.to_anndata)
      - obs contains cell metadata (cell_id, cell_type, tissue_id, microe_neighbors_count)
      - obsm contains additional multidimensional representation features (e.g., 
        raw_expression, neighbor_composition, nn_embedding) computed at the microenvironment level.

    For each MicroE, the function:
      1. Calls export_center_cell_with_representations to compute and attach the extra representations.
      2. Uses the center cell’s inherent biomarker expression to build X (as in cell.to_anndata).
      3. Extracts each requested representation from the center cell's additional_features.
         Each representation is flattened, and the arrays are padded (or trimmed) to the dimension
         determined by the first center cell that provides that representation.
      4. Constructs obs metadata for each center cell.
      5. Uses build_ann_data to create an AnnData object that preserves cell_id in obs.
         The biomarker expression matrix (X) is taken from the cell-level data, and the additional
         representations are stored in obsm under keys corresponding to the representation names.

    Returns:
      An AnnData object with:
         - X: biomarker expression matrix from the center cell.
         - obs: metadata for each center cell.
         - var: biomarker names.
         - obsm: includes "spatial" (cell positions) and additional keys for each representation.
         - uns: additional metadata (e.g. data_level, tissue_id).
    """
    if not microe_list:
        return anndata.AnnData(X=np.empty((0, 0)), obs=pd.DataFrame(), var=pd.DataFrame(), obsm={})

    cells = []
    obs_list = []
    # For each representation, record the expected dimension based on the first center cell providing it.
    expected_dims = {}
    # For storing representation features for each center cell.
    rep_obsm = {}

    # Build X_rows using the inherent biomarkers (cell-level expression) for each center cell.
    X_rows = []
    for microe in microe_list:
        # Compute and attach representation features.
        microe.export_center_cell_with_representations(
            representations=representations,
            model=model,
            device=device
        )
        c = microe.center_cell
        cells.append(c)
        
        # Use the center cell's biomarkers for X.
        bio_names = list(c.biomarkers.biomarkers.keys())
        if bio_names:
            bio_row = [c.biomarkers.biomarkers.get(bm, np.nan) for bm in bio_names]
        else:
            bio_row = []
        X_rows.append(bio_row)
        
        # Build obs metadata.
        obs_dict = {
            "tissue_id": c.tissue_id,
            "cell_id": c.cell_id,
            "cell_type": c.cell_type,
            "size": c.size,

            # additional
            "microe_neighbors_count": len(microe.neighbors)
        }
        obs_list.append(obs_dict)
        
        # Process each requested representation.
        for rep in representations:
            vec = c.get_feature(rep)
            if vec is not None:
                flat = np.ravel(vec)
            else:
                flat = np.array([])
            if rep not in expected_dims:
                expected_dims[rep] = flat.shape[0]
            target_dim = expected_dims[rep]
            # Pad with zeros if necessary.
            if flat.shape[0] < target_dim:
                flat = np.pad(flat, (0, target_dim - flat.shape[0]), constant_values=0.0)
            elif flat.shape[0] > target_dim:
                flat = flat[:target_dim]
            if rep not in rep_obsm:
                rep_obsm[rep] = []
            rep_obsm[rep].append(flat)
    
    # Determine the set of biomarkers from the first cell.
    if cells:
        biomarker_names = list(cells[0].biomarkers.biomarkers.keys())
    else:
        biomarker_names = []
    
    # Ensure X_rows are all of the same length as the biomarker set.
    max_bio = len(biomarker_names)
    padded_X_rows = []
    for row in X_rows:
        if len(row) < max_bio:
            row = row + [0.0] * (max_bio - len(row))
        elif len(row) > max_bio:
            row = row[:max_bio]
        padded_X_rows.append(row)
    X_array = np.array(padded_X_rows, dtype=float)
    
    uns = {"data_level": "center"}

    # Build AnnData using the biomarker expression (X), preserving cell_id in obs.
    adata = build_ann_data(
        cells=cells,
        X=X_array,
        extra_obs=obs_list,
        uns=uns,
        feature_names=biomarker_names
    )
    
    # Add representation features to obsm.
    for rep, rep_list in rep_obsm.items():
        # Ensure each rep_list entry is a 1D array; stack them into a 2D array (n_cells x dimension).
        rep_array = np.vstack(rep_list)
        adata.obsm[rep] = rep_array

    return adata