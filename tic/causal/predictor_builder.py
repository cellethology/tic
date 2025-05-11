# tic/causal/predictor_builder.py
from __future__ import annotations
from typing import Sequence, Literal
import numpy as np, pandas as pd, scipy.sparse as sp
from anndata import AnnData

def register_predictors(
    adata: AnnData,
    *,
    source_key: str,              # e.g. 'celltype_gene_count'
    extractor_name: str,          # e.g. 'CelltypeGeneCount'
    cell_types: Sequence[str],
    genes: Sequence[str],
    agg: Literal["mean", "sum"] = "mean",
    out_key: str = "X_predictors",
    meta_key: str = "X_predictors_meta",
    names_key: str = "X_predictors_names",
    append: bool = True,          # 允许多次追加不同 extractor
    to_sparse: bool = True,
) -> None:
    """
    • 将 `.obsm[source_key]` 统一搬运到 `.obsm[out_key]`  
    • 自动生成列名 & meta DataFrame，支持多次追加。
    """
    X_src = adata.obsm[source_key]
    if sp.issparse(X_src):
        X_src = X_src.tocsr()
    else:
        X_src = np.asarray(X_src)

    new_names = [f"{extractor_name}:{agg}_{ct}_{g}"
                 for ct in cell_types for g in genes]

    # --- 写入 / 追加矩阵 ------------------------------------------------------
    if append and out_key in adata.obsm:
        X_old = adata.obsm[out_key]
        X_concat = sp.hstack([X_old, X_src]) if sp.issparse(X_src) else np.hstack([X_old, X_src])
    else:
        X_concat = X_src

    if to_sparse and not sp.issparse(X_concat):
        X_concat = sp.csr_matrix(X_concat)

    adata.obsm[out_key] = X_concat

    # --- meta & names ---------------------------------------------------------
    meta_rows = pd.DataFrame({
        "extractor": extractor_name,
        "agg": agg,
        "cell_type": [ct for ct in cell_types for _ in genes],
        "gene": genes * len(cell_types),
        "description": "count of neighbours expressing gene"
    })
    if append and meta_key in adata.uns:
        adata.uns[meta_key] = pd.concat([adata.uns[meta_key], meta_rows], ignore_index=True)
        adata.uns[names_key] += new_names
    else:
        adata.uns[meta_key] = meta_rows
        adata.uns[names_key] = new_names