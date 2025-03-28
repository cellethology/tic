"""
tests/pseudotime/test_pseudotime.py

This module contains unit tests for the pseudotime analysis components:

1. Clustering (tic.pseudotime.clustering):
   - Tests that KMeans and AgglomerativeClustering produce cluster labels of correct shape.

2. Dimensionality Reduction (tic.pseudotime.dimensionality_reduction):
   - Tests that PCA and UMAP reduce embeddings to the specified number of components.

3. Pseudotime Inference (tic.pseudotime.pseudotime):
   - Tests that the SlingshotMethod returns pseudotime values with correct dimensions when 
     given an AnnData object with UMAP embeddings and cluster labels.

Note:
  These tests use dummy data (random embeddings) and basic checks on output shapes and types.
"""

import numpy as np
import pandas as pd
import pytest
from tic.pseudotime.clustering import Clustering
from tic.pseudotime.dimensionality_reduction import DimensionalityReduction
from tic.pseudotime.pseudotime import SlingshotMethod
import anndata

# ----------------- Tests for Clustering Module -----------------

def test_kmeans_clustering():
    # Create dummy embeddings: 50 samples, 5 features.
    embeddings = np.random.rand(50, 5)
    n_clusters = 3
    clustering = Clustering(method="kmeans", n_clusters=n_clusters, random_state=0)
    labels = clustering.cluster(embeddings)
    # Check that we get 50 labels and they are in the range [0, n_clusters-1]
    assert labels.shape[0] == 50
    assert labels.dtype.kind in 'iu'
    assert labels.min() >= 0 and labels.max() < n_clusters

def test_agg_clustering():
    embeddings = np.random.rand(50, 5)
    n_clusters = 4
    clustering = Clustering(method="agg", n_clusters=n_clusters)
    labels = clustering.cluster(embeddings)
    assert labels.shape[0] == 50
    assert labels.dtype.kind in 'iu'
    assert labels.min() >= 0 and labels.max() < n_clusters

# ----------------- Tests for Dimensionality Reduction Module -----------------

def test_pca_reduction():
    # Create dummy embeddings: 100 samples, 10 features.
    embeddings = np.random.rand(100, 10)
    n_components = 3
    dr = DimensionalityReduction(method="PCA", n_components=n_components, random_state=0)
    reduced = dr.reduce(embeddings)
    # Check that reduced embeddings has shape (100, 3)
    assert reduced.shape == (100, n_components)

def test_umap_reduction():
    embeddings = np.random.rand(100, 10)
    n_components = 2
    dr = DimensionalityReduction(method="UMAP", n_components=n_components, random_state=0)
    reduced = dr.reduce(embeddings)
    assert reduced.shape == (100, n_components)

# ----------------- Tests for Pseudotime Module (SlingshotMethod) -----------------

@pytest.fixture
def dummy_ann_data():
    """
    Create a dummy AnnData object with:
      - X: dummy data (not used by Slingshot, but required by AnnData)
      - obs: with a 'celltype' column containing cluster labels.
      - obsm: with 'X_umap' as dummy 2D embeddings.
    """
    np.random.seed(0)
    # Generate 50 samples with 2D embeddings.
    embeddings = np.random.rand(50, 2)
    # Dummy cluster labels: integers between 0 and 1 (for 2 clusters).
    labels = np.random.randint(0, 2, 50)
    obs = pd.DataFrame({"celltype": labels}, index=[str(i) for i in range(50)])
    obsm = {"X_umap": embeddings}
    # X can be same as embeddings for simplicity.
    X = embeddings.copy()
    adata = anndata.AnnData(X=X, obs=obs, obsm=obsm)
    return adata

def test_slingshot_with_start_node(dummy_ann_data, tmp_path):
    """
    Test that SlingshotMethod returns a pseudotime vector when a start_node is provided.
    """
    # Use a fixed start node.
    start_node = 0
    slingshot = SlingshotMethod(start_node=start_node)
    # Run pseudotime analysis; we use a temporary directory for plots.
    pseudotime = slingshot.analyze(
        labels=dummy_ann_data.obs["celltype"].to_numpy(),
        umap_embeddings=dummy_ann_data.obsm["X_umap"],
        output_dir=str(tmp_path)
    )
    # Check that pseudotime is a 1D array of length equal to the number of cells.
    assert isinstance(pseudotime, np.ndarray)
    assert pseudotime.ndim == 1
    assert pseudotime.shape[0] == dummy_ann_data.n_obs

def test_slingshot_without_start_node(dummy_ann_data, tmp_path):
    """
    Test that SlingshotMethod still returns a pseudotime vector when no start_node is provided.
    For this test, we simply set start_node=None.
    """
    slingshot = SlingshotMethod(start_node=None)
    pseudotime = slingshot.analyze(
        labels=dummy_ann_data.obs["celltype"].to_numpy(),
        umap_embeddings=dummy_ann_data.obsm["X_umap"],
        output_dir=str(tmp_path)
    )
    assert isinstance(pseudotime, np.ndarray)
    assert pseudotime.ndim == 1
    assert pseudotime.shape[0] == dummy_ann_data.n_obs