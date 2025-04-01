#!/usr/bin/env python3
"""
Unit tests for the high-level causal inference pipeline (tic.pipeline.causal).

Tested functionalities:
1. Validation of required keys in the AnnData object.
2. Construction of a time-series DataFrame from an AnnData object using the real utility function.
3. Wrapping of the time-series data into CausalInput objects.
4. Obtaining and using a causal method (via the factory) to fit the model and estimate effect.
5. Storage of causal inference results in adata.uns["causal_results"].
"""

import pytest
import numpy as np
import anndata
import pandas as pd

from tic.pipeline.causal import run_causal_pipeline
from tic.causal.factory import CausalMethodFactory


class DummyCausalMethod:
    """Dummy causal method for testing purposes."""
    def fit(self, ci, **kwargs):
        self.ci = ci

    def estimate_effect(self, ci, **kwargs):
        return {"dummy": "result"}


@pytest.fixture
def dummy_ann_data():
    """
    Create a minimal AnnData object with the required keys:
      - obs: includes a 'pseudotime' column.
      - var: includes an index with the outcome biomarker "PanCK".
      - X: a numpy array with the center cell's biomarker expression (for outcome).
      - obsm: includes "neighbor_biomarker" as a 2D array.
      - uns: includes "neighbor_biomarker_feature_names" as a list.
    
    All indices are set as strings to avoid index mismatch errors.
    """
    index = ["0", "1", "2"]
    # obs with pseudotime.
    obs = pd.DataFrame({"pseudotime": [0.1, 1.2, 2.3]}, index=index)
    # var with outcome biomarker "PanCK".
    var = pd.DataFrame(index=["PanCK"])
    # X is a (3, 1) numpy array for outcome biomarker.
    X = np.array([[100], [200], [300]])
    # obsm contains a neighbor biomarker matrix with 2 predictors.
    neighbor_biomarker = np.array([[10, 20],
                                   [30, 40],
                                   [50, 60]])
    obsm = {"neighbor_biomarker": neighbor_biomarker}
    # uns includes a list mapping neighbor biomarker columns to predictor names.
    uns = {"neighbor_biomarker_feature_names": ["X1", "X2"]}
    return anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm, uns=uns)


def test_run_causal_pipeline(monkeypatch, dummy_ann_data):
    """
    Test the high-level causal inference pipeline.
    
    This test verifies:
      - The pipeline correctly constructs a time-series DataFrame from AnnData.
      - The pipeline wraps the data into CausalInput objects.
      - The pipeline uses a causal method (via the factory) to fit and 
        estimate effect.
      - The pipeline stores the resulting outputs in adata.uns["causal_results"].
    """
    # Monkeypatch CausalMethodFactory.get_method to always return DummyCausalMethod.
    monkeypatch.setattr(CausalMethodFactory, "get_method", lambda method_name, **kwargs: DummyCausalMethod())
    
    # Call the pipeline with the dummy AnnData.
    # Use outcome biomarker "PanCK" (which is in dummy_ann_data.var.index).
    adata_result = run_causal_pipeline(
        dummy_ann_data,
        y_biomarker="PanCK",
        x_variable=None,            # Use all predictors except "pseudotime" and "Y".
        pseudotime_key="pseudotime",
        causal_method="dummy",      # This value is ignored because of the monkeypatch.
        copy=True,
        bins=None,
        causal_kwargs={"test_param": 123},
        method_kwargs={"init_param": "abc"}
    )

    # Verify that a new AnnData object is returned (copy=True).
    assert adata_result is not dummy_ann_data

    # Verify that the pipeline stored results in adata.uns["causal_results"].
    assert "causal_results" in adata_result.uns
    causal_results = adata_result.uns["causal_results"]

    # The construct_time_series function (in tic.causal.utils) builds a DataFrame with:
    #  - "pseudotime" from adata.obs,
    #  - "Y" from adata.X (outcome biomarker "PanCK"),
    #  - and predictor columns named from uns["neighbor_biomarker_feature_names"].
    #
    # Since x_variable is None, the predictors are all columns except "pseudotime" and "Y",
    # which should be ["X1", "X2"].
    assert "X1" in causal_results
    assert "X2" in causal_results

    # Each predictor should have been processed by the dummy causal method.
    expected_result = {"dummy": "result"}
    assert causal_results["X1"] == expected_result
    assert causal_results["X2"] == expected_result