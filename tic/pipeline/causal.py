from typing import List, Optional, Dict, Any, Union
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
      1. Assumes pseudotime inference has been performed so that adata.obs[pseudotime_key] exists.
      2. Constructs a time-series DataFrame that includes:
             - "pseudotime": from adata.obs[pseudotime_key]
             - Outcome variable (Y): the center cell's expression of the specified biomarker (y_biomarker)
             - Predictor variables: extracted from the flattened neighbor biomarker matrix.
                 If x_variable is:
                    - a string: that column is used;
                    - a list: only those columns are used;
                    - None: iterate over all predictor columns (i.e. all columns except "pseudotime" and "Y"),
                      and run separate causal inferences for each predictor.
      3. Wraps each time-series DataFrame into a CausalInput object.
      4. Retrieves the desired causal inference method from the factory and runs its fit and estimate_effect steps.
      5. Stores the causal results in adata.uns["causal_results"].
    
    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object that must include:
          - obs[pseudotime_key]: pseudotime values for each center cell.
          - X: biomarker expression matrix for center cells.
          - obsm["neighbor_biomarker"]: flattened neighbor biomarker matrix.
          - uns["neighbor_biomarker_feature_names"]: mapping of neighbor biomarker columns.
    y_biomarker : str
        The outcome biomarker (e.g., "PanCK") for which the time series is constructed.
    x_variable : list of str, or None
        The predictor variable(s) from the neighbor biomarker matrix.
        If None, the pipeline will iterate over all predictor columns (i.e., all columns in the
        time-series DataFrame except "pseudotime" and "Y").
    pseudotime_key : str, optional
        Key in adata.obs that stores pseudotime values. Default is "pseudotime".
    causal_method : str, optional
        The identifier for the causal method to use (default "granger_causality").
    copy : bool, optional
        If True, operate on a copy of adata and return it; if False, modify adata in place and return None.
    bins : int, optional
        Number of bins for grouping the time-series data (averaging within each bin). If None, no binning is performed.
    causal_kwargs : dict, optional
        Additional keyword arguments to pass to the causal inference method.
    
    Returns
    -------
    Optional[anndata.AnnData]
        If copy is True, returns the modified AnnData object with causal results stored in uns["causal_results"].
        If copy is False, returns None.
    
    Raises
    ------
    ValueError
        If required keys (pseudotime, neighbor_biomarker, or corresponding feature names) are missing,
        or if the selected predictor variable(s) are not found.
    """
    if causal_kwargs is None:
        causal_kwargs = {}

    if copy:
        adata = adata.copy()

    # Check that required keys exist in adata.
    if pseudotime_key not in adata.obs.columns:
        raise ValueError(f"AnnData.obs must contain the '{pseudotime_key}' column.")
    if "neighbor_biomarker" not in adata.obsm:
        raise ValueError("AnnData.obsm must contain 'neighbor_biomarker'.")
    if "neighbor_biomarker_feature_names" not in adata.uns:
        raise ValueError("AnnData.uns must contain 'neighbor_biomarker_feature_names'.")

    # Step 1: Construct time-series DataFrame.
    ts_df = construct_time_series(adata, y_biomarker, bins)
    
    # Determine predictor columns.
    if x_variable is None:
        # Use all columns except "pseudotime" and "Y".
        predictors = [col for col in ts_df.columns if col not in ["pseudotime", "Y"]]
        # For each predictor, run causal inference separately.
        causal_results = {}
        for pred in predictors:
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
    elif isinstance(x_variable, list):
        predictors = x_variable
        causal_results = {}
        for pred in predictors:
            if pred not in ts_df.columns:
                raise ValueError(f"Predictor variable '{pred}' not found in the constructed time series data.")
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

    # Store the causal results in adata.uns.
    adata.uns["causal_results"] = causal_results

    return adata if copy else None