# tests/pipeline/test_pipeline.py

"""
Unit Tests for Pseudotime and Causal Inference Pipelines

This file contains unit tests for the following functionalities:
1. Pseudotime Pipeline:
   - Verify that the dimensionality reduction, clustering, and pseudotime inference steps are correctly applied.
   - Ensure the output AnnData object includes the fields 'rp_reduced', 'cluster', and 'pseudotime' with expected shapes and values.

2. Causal Inference Pipeline:
   - Test the construction of the time-series DataFrame from AnnData.
   - Validate that the causal inference pipeline generates causal results correctly.
   - Confirm that the causal results are stored in adata.uns['causal_results'].
   
The tests utilize monkeypatching to replace actual implementations with dummy functions to ensure predictable and isolated testing.
"""

import numpy as np
import pandas as pd
import anndata
import pytest

from tic.pipeline.pseudotime import run_pseudotime_pipeline
from tic.pipeline.causal import run_causal_pipeline

# ---------------------------
# Dummy implementations for pseudotime pipeline
# ---------------------------
class DummyDR:
    def __init__(self, method, n_components):
        pass

    def reduce(self, embeddings):
        # Return first n_components columns from each embedding vector
        n_components = 2
        return np.array([emb[:n_components] for emb in embeddings])

class DummyClustering:
    def __init__(self, method, n_clusters):
        self.n_clusters = n_clusters

    def cluster(self, embeddings):
        n = embeddings.shape[0]
        # Return cyclic cluster labels based on n_clusters
        return np.array([i % self.n_clusters for i in range(n)])

class DummySlingshot:
    def __init__(self, start_node):
        self.start_node = start_node

    def analyze(self, cluster_labels, reduced_embeddings, output_dir=""):
        n = len(cluster_labels)
        # Return a linearly spaced pseudotime vector from 0 to 1
        return np.linspace(0, 1, n)

# ---------------------------
# Dummy implementations for causal inference pipeline
# ---------------------------
class DummyCausalMethod:
    def fit(self, ci, **kwargs):
        pass

    def estimate_effect(self, ci, **kwargs):
        # Return a fixed result dictionary
        return {"effect": 0.5}

def dummy_construct_time_series(adata, y_biomarker, bins):
    # Build a simple DataFrame with pseudotime, outcome 'Y', and one predictor 'predictor1'
    df = pd.DataFrame({
        "pseudotime": adata.obs["pseudotime"],
        "Y": adata.X[:, adata.var_names.get_loc(y_biomarker)],
        "predictor1": np.random.rand(adata.n_obs)
    })
    return df

class DummyCausalMethodFactory:
    @staticmethod
    def get_method(method):
        return DummyCausalMethod()

# ---------------------------
# Apply monkeypatching to replace actual implementations
# ---------------------------
@pytest.fixture(autouse=True)
def patch_pseudotime(monkeypatch):
    monkeypatch.setattr("tic.pipeline.pseudotime.DimensionalityReduction", DummyDR)
    monkeypatch.setattr("tic.pipeline.pseudotime.Clustering", DummyClustering)
    monkeypatch.setattr("tic.pipeline.pseudotime.SlingshotMethod", DummySlingshot)

@pytest.fixture(autouse=True)
def patch_causal(monkeypatch):
    monkeypatch.setattr("tic.pipeline.causal.construct_time_series", dummy_construct_time_series)
    monkeypatch.setattr("tic.pipeline.causal.CausalMethodFactory", DummyCausalMethodFactory)

# ---------------------------
# Unit test for pseudotime pipeline
# ---------------------------
def test_run_pseudotime_pipeline():
    # Create a simple AnnData object with 5 cells and 4 features
    data = np.random.rand(5, 4)
    adata = anndata.AnnData(X=data)
    # Store the raw expression data in obsm under 'raw_expression'
    adata.obsm["raw_expression"] = data
    adata.obs = pd.DataFrame(index=[f"cell{i}" for i in range(5)])
    
    # Run the pseudotime pipeline
    result = run_pseudotime_pipeline(adata, copy=True)
    
    # Check that expected fields are added to the AnnData object
    assert "rp_reduced" in result.obsm, "Missing reduced embeddings 'rp_reduced'"
    assert "cluster" in result.obs.columns, "Missing cluster labels in 'cluster'"
    assert "pseudotime" in result.obs.columns, "Missing pseudotime values in 'pseudotime'"
    
    # Verify the shape of reduced embeddings (should be 5 x 2)
    assert result.obsm["rp_reduced"].shape == (5, 2), "Reduced embeddings shape is incorrect"
    # Verify that the cluster labels and pseudotime arrays have the correct length
    assert len(result.obs["cluster"]) == 5, "Number of cluster labels does not match number of cells"
    assert len(result.obs["pseudotime"]) == 5, "Number of pseudotime values does not match number of cells"

# ---------------------------
# Unit test for causal inference pipeline
# ---------------------------
def test_run_causal_pipeline():
    # Create a simple AnnData object with 5 cells and 2 features
    data = np.random.rand(5, 2)
    var_names = ["PanCK", "Other"]
    adata = anndata.AnnData(X=data)
    adata.var_names = var_names
    # Add pseudotime information to obs
    adata.obs["pseudotime"] = np.linspace(0, 1, 5)
    # Add a dummy neighbor biomarker matrix to obsm
    adata.obsm["neighbor_biomarker"] = np.random.rand(5, 3)
    # Add neighbor biomarker feature names to uns
    adata.uns["neighbor_biomarker_feature_names"] = ["predictor1", "predictor2", "predictor3"]
    
    # Run the causal inference pipeline
    result = run_causal_pipeline(adata, y_biomarker="PanCK", copy=True)
    
    # Check that causal results are stored in adata.uns
    assert "causal_results" in result.uns, "Missing causal results in 'causal_results'"
    causal_results = result.uns["causal_results"]
    
    # When x_variable is None, the dummy time-series only includes 'predictor1' as predictor
    assert "predictor1" in causal_results, "predictor1 not found in causal results"
    # Verify that the causal result for predictor1 contains an 'effect' field
    assert "effect" in causal_results["predictor1"], "Missing 'effect' in the causal result for predictor1"