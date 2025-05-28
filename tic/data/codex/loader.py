"""tic.codex.loader
===================

Load pre-downloaded Murine CODEX datasets into `anndata.AnnData` objects.

Functions
---------
- list_regions: Enumerate spatial region identifiers in a CODEX dataset.
- load_region: Load expression and metadata for a single CODEX region.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, List, Union

import numpy as np
import pandas as pd
from anndata import AnnData

from ...constant import DEFAULT_DATACACHE_DIR
from .download import ensure_codex_dataset
from ..utils import check_spatial_anndata

logger = __import__('logging').getLogger(__name__)
logger.addHandler(__import__('logging').StreamHandler())
logger.setLevel(__import__('logging').INFO)

PathLike = Union[str, Path]


def list_regions(
    dataset: Literal["upmc", "charville", "dfci"],
    cache_dir: PathLike = DEFAULT_DATACACHE_DIR,
) -> List[str]:
    """
    List all region IDs available within a CODEX dataset.

    Parameters
    ----------
    dataset
        One of 'upmc', 'charville', or 'dfci'.
    cache_dir
        Root cache directory where datasets are stored.

    Returns
    -------
    List[str]
        Sorted list of region identifiers (filenames without extension).

    Example
    -------
    >>> list_regions('upmc')
    ['UPMC_c001_v001_r001_reg001', 'UPMC_c001_v001_r001_reg002', ...]
    """
    root = ensure_codex_dataset(dataset, cache_dir)
    pattern = '*.cell_data.csv'
    return sorted(p.stem for p in Path(root).glob(pattern))


def load_region(
    dataset: Literal["upmc", "charville", "dfci"],
    region_id: str,
    cache_dir: PathLike = DEFAULT_DATACACHE_DIR,
) -> AnnData:
    """
    Load a single spatial region from a CODEX dataset into AnnData.

    Parameters
    ----------
    dataset
        CODEX dataset name ('upmc', 'charville', or 'dfci').
    region_id
        Region identifier, e.g. 'UPMC_c001_v001_r001_reg001'.
    cache_dir
        Root cache directory where datasets are stored.

    Returns
    -------
    AnnData
        - .X: cell-by-biomarker expression matrix (float32)
        - .obs: DataFrame with columns:
            'cell_id', 'cell_type', 'size'
        - .var: DataFrame indexed by biomarker names
        - .obsm['spatial']: 2D coordinates as numpy array
        - .uns: metadata with 'tissue_id' and 'data_level'
    """
    root = ensure_codex_dataset(dataset, cache_dir)

    # Define file paths
    files = {
        'coords': Path(root) / f"{region_id}.cell_data.csv",
        'features': Path(root) / f"{region_id}.cell_features.csv",
        'types': Path(root) / f"{region_id}.cell_types.csv",
        'expr': Path(root) / f"{region_id}.expression.csv",
    }
    # Read tables
    dfs = {k: pd.read_csv(p) for k, p in files.items()}

    # Normalize cell IDs to strings
    for df in dfs.values():
        df['CELL_ID'] = df['CELL_ID'].astype(str)

    # Drop acquisition ID if present
    expr = dfs['expr']
    if 'ACQUISITION_ID' in expr.columns:
        expr = expr.drop(columns=['ACQUISITION_ID'])

    # Merge all dataframes on CELL_ID
    merged = (
        dfs['coords']
        .merge(dfs['features'], on='CELL_ID')
        .merge(dfs['types'], on='CELL_ID')
        .merge(expr, on='CELL_ID')
    )

    # Biomarker columns
    biom_cols = [col for col in expr.columns if col != 'CELL_ID']
    X = merged[biom_cols].to_numpy(dtype=np.float32)

    # Build AnnData
    obs = (
        merged[['CELL_ID', 'CELL_TYPE', 'SIZE']]
        .rename(columns={
            'CELL_ID': 'cell_id',
            'CELL_TYPE': 'cell_type',
            'SIZE': 'size'
        })
        .set_index('cell_id')
    )
    var = pd.DataFrame(index=biom_cols)
    obsm = {'spatial': merged[['X', 'Y']].to_numpy(dtype=np.float32)}

    adata = AnnData(X=X, obs=obs, var=var, obsm=obsm)
    adata.uns['tissue_id'] = region_id
    adata.uns['data_level'] = 'tissue'
    adata = check_spatial_anndata(adata)
    return adata
