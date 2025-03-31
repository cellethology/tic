"""
Module: tic.causal.utils

Provides utility functions for causal inference, including constructing
a time series DataFrame from an AnnData object.
"""

from typing import Optional
import pandas as pd
import anndata


def construct_time_series(
    adata: anndata.AnnData,
    y_biomarker: str,
    bin: Optional[int] = None
) -> pd.DataFrame:
    """
    Construct a time series DataFrame for causal inference from an AnnData object.

    The AnnData object is expected to include:
      - obs["pseudotime"]: Pseudotime values for each center cell.
      - X: The center cell's biomarker expression vector (used to extract outcome variable Y).
      - obsm["neighbor_biomarker"]: A 2D array where each row is the flattened neighbor biomarker matrix.
      - uns["neighbor_biomarker_feature_names"]: A list mapping each column in the neighbor matrix
        to a (cell type & biomarker) identifier.

    The resulting DataFrame will include:
      - A "pseudotime" column.
      - A "Y" column representing the outcome biomarker.
      - Predictor columns with names from uns["neighbor_biomarker_feature_names"].
      - Rows sorted in ascending order by pseudotime.

    If `bin` is provided, the DataFrame rows are grouped into the specified number of bins
    based on pseudotime and the values averaged within each bin.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object produced by export_center_cells and pseudotime inference.
    y_biomarker : str
        The biomarker to be used as the outcome variable (e.g., "PanCK").
    bin : Optional[int], optional
        Number of bins to group pseudotime values, by default None.

    Returns
    -------
    pd.DataFrame
        A time series DataFrame for causal inference. If bin is provided, each row corresponds
        to the average values within that pseudotime bin.

    Raises
    ------
    ValueError
        If required keys are missing in the AnnData object.
    """
    # Validate required keys in the AnnData object.
    if "pseudotime" not in adata.obs.columns:
        raise ValueError("AnnData.obs must contain a 'pseudotime' column.")
    if "neighbor_biomarker" not in adata.obsm:
        raise ValueError("AnnData.obsm must contain 'neighbor_biomarker'.")
    if "neighbor_biomarker_feature_names" not in adata.uns:
        raise ValueError("AnnData.uns must contain 'neighbor_biomarker_feature_names'.")

    # Extract pseudotime values.
    pseudotime = adata.obs["pseudotime"].values

    # Extract outcome (Y) from the center cell's biomarker expression.
    biomarker_names = list(adata.var.index)
    if y_biomarker not in biomarker_names:
        raise ValueError(f"Outcome biomarker '{y_biomarker}' not found in adata.var.index.")
    y_idx = biomarker_names.index(y_biomarker)
    Y = adata.X[:, y_idx].flatten()

    # Extract predictor variables from the flattened neighbor biomarker matrix.
    X_neighbor = adata.obsm["neighbor_biomarker"]
    predictor_names = adata.uns["neighbor_biomarker_feature_names"]

    # Construct the time series DataFrame.
    df = pd.DataFrame({
        "pseudotime": pseudotime,
        "Y": Y
    }, index=adata.obs.index)
    predictors_df = pd.DataFrame(X_neighbor, index=adata.obs.index, columns=predictor_names)
    df = pd.concat([df, predictors_df], axis=1)

    # Sort DataFrame by pseudotime.
    df.sort_values("pseudotime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # If binning is requested, group rows into bins and average values.
    if bin is not None:
        df["bin"] = pd.cut(df["pseudotime"], bins=bin)
        df = df.groupby("bin").mean().reset_index(drop=True)

    return df