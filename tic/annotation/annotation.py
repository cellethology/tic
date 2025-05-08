# file: tic/annotation/annotation.py
from anndata import AnnData
import scanpy as sc

def assign_cell_clusters(
    h5ad_path: str | None = None,
    adata: AnnData | None = None,
    min_cells: int = 0,
    min_genes: int = 0,
    n_pca: int = 50,
    n_neighbors: int = 15,
    resolution: float = 2.0,
    n_filter_highly_variable_genes: int = 2000,
    copy: bool = False,
) -> AnnData:
    """
    End-to-end pipeline for clustering cells using Scanpy (up to Leiden clustering and marker gene ranking).

    Parameters
    ----------
    h5ad_path : str | None
        Path to the .h5ad file. Optional if `adata` is provided.
    adata : AnnData | None
        AnnData object containing single-cell expression data.
    min_cells : int
        Minimum number of cells a gene must be expressed in.
    min_genes : int
        Minimum number of genes a cell must express.
    n_pca : int
        Number of principal components to compute.
    n_neighbors : int
        Number of neighbors for the KNN graph.
    resolution : float
        Resolution parameter for Leiden clustering. 
    n_filter_highly_variable_genes : int
        Number of highly variable genes to select.
    copy : bool
        Whether to return a copy of the AnnData object.

    Returns
    -------
    AnnData
        Processed AnnData object with clustering and UMAP.
    """
    if h5ad_path is not None:
        adata = sc.read_h5ad(h5ad_path)
    elif adata is None:
        raise ValueError("Either `h5ad_path` or `adata` must be provided.")

    if copy:
        adata = adata.copy()

    # Filter cells and genes
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Normalize and log transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Filter highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_filter_highly_variable_genes, subset=True)

    # PCA
    sc.pp.pca(adata, n_comps=n_pca)

    # Build neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pca)

    # Leiden clustering
    sc.tl.leiden(adata, resolution=resolution)

    # UMAP
    sc.tl.umap(adata)

    return adata

def return_top_genes(
    adata: AnnData,
    groupby: str = "leiden",
    rank_genes_groups_method: str = "t-test",
    n_genes: int = 5,
) -> dict:
    """
    Return the top n marker genes for each cluster.

    Returns a dictionary of cluster_id -> list of top n marker genes and scores:
    {
        'cluster_id': {
            'genes': list of top n marker genes,
            'scores': list of scores
        }
    }

    Parameters
    ----------
    adata : AnnData
        The AnnData object.
    groupby : str
        The column name of the cluster labels.
    rank_genes_groups_method : str
        The method to rank the marker genes.
    n_genes : int
        The number of top marker genes to return.

    Returns
    -------
    dict
        A dictionary of cluster_id -> list of top n marker genes and scores.
    """
    sc.tl.rank_genes_groups(adata, groupby=groupby, method=rank_genes_groups_method, n_genes=n_genes)
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names  # get all cluster ids
    
    top_genes = {}
    for cluster_id in groups:
        top_genes[cluster_id] = {
            'genes': result['names'][cluster_id][:n_genes].tolist(),
            'scores': result['scores'][cluster_id][:n_genes].tolist()
        }
    return top_genes

def transform_cluster_annotation(cluster_annotation: dict) -> dict:
    """
    from: 
    {
        "cluster_id": {
            "assigned_cell_type": "Cell Type Name",
            "reasoning": "Brief justification based on marker gene expression."
        },
        ...
    }
    to:
    {
        "cluster_id": "Cell Type Name",
        ...
    }
    """
    cluster_cell_type_mapping = {}
    for cluster_id, cell_type in cluster_annotation.items():
        cluster_cell_type_mapping[cluster_id] = cell_type["assigned_cell_type"]
    return cluster_cell_type_mapping
