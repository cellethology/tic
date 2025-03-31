"""
Module: tic.pipeline.causal

Runs the high-level causal inference pipeline on an AnnData object.
This pipeline requires that pseudotime inference has already been performed.
It constructs a time-series DataFrame, wraps it into CausalInput objects,
runs the selected causal inference method, and stores the results in adata.uns.
"""

from typing import List, Optional, Dict, Any
import anndata

from tic.causal.causal_input import CausalInput
from tic.causal.factory import CausalMethodFactory
from tic.causal.utils import construct_time_series


def run_causal_pipeline(
    adata: anndata.AnnData,
    y_biomarker: str,
    x_variable: Optional[List[str]] = None,
    pseudotime_key: str = "pseudotime",
    causal_method: str = "granger_causality",
    copy: bool = True,
    bins: Optional[int] = None,
    causal_kwargs: Optional[Dict[str, Any]] = None
) -> Optional[anndata.AnnData]:
    """
    Run a high-level causal inference pipeline on an AnnData object that already contains pseudotime values.

    The pipeline performs the following steps:
      1. Verify that adata.obs contains pseudotime and that adata.obsm/uns have the required keys.
      2. Construct a time-series DataFrame with:
         - "pseudotime": from adata.obs[pseudotime_key]
         - "Y": the center cell's expression for y_biomarker
         - Predictor variables extracted from obsm["neighbor_biomarker"]
      3. For each predictor column (or the ones specified in x_variable), wrap the data into a CausalInput,
         then use the causal method (from CausalMethodFactory) to fit and estimate effect.
      4. Store the causal results in adata.uns["causal_results"].

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object that must include:
          - obs[pseudotime_key]: pseudotime values.
          - X: center cell biomarker expression matrix.
          - obsm["neighbor_biomarker"]: flattened neighbor biomarker matrix.
          - uns["neighbor_biomarker_feature_names"]: mapping for neighbor biomarker columns.
    y_biomarker : str
        Outcome biomarker (e.g., "PanCK") used to build the time series.
    x_variable : Optional[List[str]], optional
        Predictor variable(s) to use. If None, all columns except "pseudotime" and "Y" are used.
    pseudotime_key : str, optional
        Key in adata.obs for pseudotime values. Default is "pseudotime".
    causal_method : str, optional
        Identifier for the causal method to use (default "granger_causality").
    copy : bool, optional
        If True, work on a copy of adata; otherwise, modify adata in place.
    bins : Optional[int], optional
        Number of bins for grouping time-series data (averaging values within each bin).
    causal_kwargs : Optional[Dict[str, Any]], optional
        Additional keyword arguments to pass to the causal inference method.

    Returns
    -------
    Optional[anndata.AnnData]
        If copy is True, returns the modified AnnData with causal results stored in uns["causal_results"].
        If copy is False, modifies adata in place and returns None.

    Raises
    ------
    ValueError
        If required keys are missing or if specified predictor variables are not found.
    """
    if causal_kwargs is None:
        causal_kwargs = {}

    if copy:
        adata = adata.copy()

    # Validate required keys.
    if pseudotime_key not in adata.obs.columns:
        raise ValueError(f"AnnData.obs must contain the '{pseudotime_key}' column.")
    if "neighbor_biomarker" not in adata.obsm:
        raise ValueError("AnnData.obsm must contain 'neighbor_biomarker'.")
    if "neighbor_biomarker_feature_names" not in adata.uns:
        raise ValueError("AnnData.uns must contain 'neighbor_biomarker_feature_names'.")

    # Step 1: Construct the time-series DataFrame.
    ts_df = construct_time_series(adata, y_biomarker, bins)

    # Step 2: Determine predictor columns and run causal inference.
    causal_results: Dict[str, Any] = {}
    if x_variable is None:
        # Use all columns except "pseudotime" and "Y".
        predictors = [col for col in ts_df.columns if col not in ["pseudotime", "Y"]]
    else:
        predictors = x_variable

    for pred in predictors:
        if pred not in ts_df.columns:
            raise ValueError(f"Predictor variable '{pred}' not found in the time-series data.")
        ci = CausalInput(
            data=ts_df,
            treatment_col=pred,
            outcome_col="Y",
            covariates=[],
            extra_params={}
        )
        causal_method_obj = CausalMethodFactory.get_method(causal_method)
        causal_method_obj.fit(ci, **causal_kwargs)
        result = causal_method_obj.estimate_effect(ci, **causal_kwargs)
        causal_results[pred] = result

    # Step 3: Store results in adata.uns.
    adata.uns["causal_results"] = causal_results

    return adata if copy else None