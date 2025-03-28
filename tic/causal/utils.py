import pandas as pd
import numpy as np
import anndata
from typing import Optional

def construct_time_series(
    adata: anndata.AnnData,
    y_biomarker: str,
    bin: Optional[int] = None
) -> pd.DataFrame:
    """
    Construct a time series DataFrame for causal inference from an AnnData object.

    The AnnData is expected to include:
      - obs["pseudotime"]: The pseudotime values for each center cell.
      - X: Center cell biomarker expression vector (used to extract outcome variable Y).
      - obsm["neighbor_biomarker"]: A 2D array where each row is the flattened neighbor biomarker matrix.
      - uns["neighbor_biomarker_feature_names"]: A list mapping each column in the neighbor matrix 
        to a (cell type & biomarker) identifier.
    
    The output DataFrame will include:
      - A "pseudotime" column.
      - A "Y" column representing the center cell's expression of the specified biomarker.
      - Predictor columns with names as given in uns["neighbor_biomarker_feature_names"].
      - Rows are sorted in ascending order by pseudotime.
    
    If the parameter `bin` is provided, the function groups rows into the specified number of bins 
    based on pseudotime and returns the average value for each bin.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object output from export_center_cells and pseudotime inference.
    y_biomarker : str
        The biomarker (e.g., "PanCK") to be used as the outcome variable.
    bin : int or None, optional
        If provided, group pseudotime values into this many bins and average the values within each bin.
    
    Returns
    -------
    pd.DataFrame
        A time series DataFrame for causal inference. If bin is provided, each row corresponds to 
        the average values within that pseudotime bin.
    """
    # Check required keys
    if "pseudotime" not in adata.obs.columns:
        raise ValueError("AnnData.obs must contain a 'pseudotime' column.")
    if "neighbor_biomarker" not in adata.obsm:
        raise ValueError("AnnData.obsm must contain 'neighbor_biomarker'.")
    if "neighbor_biomarker_feature_names" not in adata.uns:
        raise ValueError("AnnData.uns must contain 'neighbor_biomarker_feature_names'.")

    # Extract pseudotime.
    pseudotime = adata.obs["pseudotime"].values

    # Extract outcome (Y) from the center cell's biomarker expression.
    biomarker_names = list(adata.var.index)
    if y_biomarker not in biomarker_names:
        raise ValueError(f"Outcome biomarker '{y_biomarker}' not found in adata.var.index.")
    y_idx = biomarker_names.index(y_biomarker)
    Y = adata.X[:, y_idx].flatten()

    # Extract predictor variables from the flattened neighbor biomarker matrix.
    X_neighbor = adata.obsm["neighbor_biomarker"]  # shape: (n_cells, n_features_neighbor)
    predictor_names = adata.uns["neighbor_biomarker_feature_names"]

    # Construct the DataFrame.
    df = pd.DataFrame({
        "pseudotime": pseudotime,
        "Y": Y
    }, index=adata.obs.index)
    predictors_df = pd.DataFrame(X_neighbor, index=adata.obs.index, columns=predictor_names)
    df = pd.concat([df, predictors_df], axis=1)

    # Sort DataFrame by pseudotime.
    df.sort_values("pseudotime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # If bin is provided, group data into bins and average within each bin.
    if bin is not None:
        df["bin"] = pd.cut(df["pseudotime"], bins=bin)
        df_binned = df.groupby("bin").mean().reset_index(drop=True)
        df = df_binned

    return df