from typing import List
from anndata import AnnData
import scipy

from tic.constant import EMT_GENES, EMT_TF, EPITHELIAL_GENES, MESENCHYMAL_GENES, DEFAULT_KEY

def check_spatial_anndata(adata: AnnData) -> AnnData:
    '''
    check if input adata meet minimum requirements:
        1. X stores genes/biomarker expressions
        2. var_names should store gene/biomarker names
        3. obsm['spatial'] stores 2D spatial coordinates
    '''
    if adata.X is None:
        raise ValueError("X must be present in adata")
    if adata.var_names is None:
        raise ValueError("var_names must be present in adata")
    if 'spatial' not in adata.obsm:
        raise ValueError("spatial must be present in adata.obsm")
    if adata.obsm['spatial'].shape[1] != 2:
        raise ValueError("spatial must be a 2D array")
    if adata.var_names.isnull().any():
        raise ValueError("var_names must not contain null values")
    if adata.var_names.duplicated().any():
        raise ValueError("var_names must not contain duplicated values")

    # transform X into numpy array if it is a sparse matrix
    if isinstance(adata.X, scipy.sparse.csr_matrix):
        adata.X = adata.X.toarray()

    return adata

def check_EMT_genes(adata: AnnData) -> AnnData:
    '''
    check if the EMT genes are present in the adata
    '''
    if not any(gene in adata.var_names for gene in EMT_GENES):
        raise ValueError("EMT genes must be present in adata.var_names")
    
    print(f"Present Epithelial genes: {list(set(EPITHELIAL_GENES) & set(adata.var_names))}")
    print(f"Present Mesenchymal genes: {list(set(MESENCHYMAL_GENES) & set(adata.var_names))}")
    print(f"Present EMT TF genes: {list(set(EMT_TF) & set(adata.var_names))}")
    return adata

def get_cell_types(adata: AnnData) -> List[str]:
    """
    Return the ordered list of cell types to use in feature extractors.

    Priority:
      1. adata.obs['cell_type'].cat.categories  (if column exists & is categorical)
      2. adata.obs['cell_type'].unique()         (fallback if not categorical)
      3. ['Unassigned']                          (if no cell type column)
    """
    # 0) category in uns
    if DEFAULT_KEY.get('cell_types') in adata.uns:
        col = adata.uns[DEFAULT_KEY.get('cell_types')]
        if isinstance(col, list):
            return list(map(str, col))  
        elif hasattr(col, "cat"):
            return list(col.cat.categories)
        else:
            return list(col.astype(str).unique())

    # 1) category in obs
    if DEFAULT_KEY.get('cell_type') in adata.obs:
        col = adata.obs[DEFAULT_KEY.get('cell_type')]
        if hasattr(col, "cat"):
            return list(col.cat.categories)
        else:
            return list(col.astype(str).unique())

    # 2) fallback
    adata.obs["cell_type"] = "Unassigned"
    return ["Unassigned"]


def get_biomarkers(adata: AnnData) -> List[str]:
    """
    Return the ordered list of biomarkers (genes) to use in feature extractors.
    
    Priority:
      1. adata.var_names
      2. index-based names ['Var_0', 'Var_1', ...]
    """    
    if adata.var_names is not None:
        return list(adata.var_names)
    
    # fallback if var_names missing
    n = adata.X.shape[1] if hasattr(adata, "X") else 0
    return [f"Var_{i}" for i in range(n)]