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

    Returns
    -------
    List[str]
        Sorted list of region identifiers like 'UPMC_c001_v001_r001_reg001'
    """
    root = ensure_codex_dataset(dataset, cache_dir)
    pattern = '*.cell_data.csv'
    return sorted(
        p.name.split(".cell_data")[0]
        for p in Path(root).glob(pattern)
    )

def _find_file(root: Path, region_id: str, suffix: str) -> Path:
    """
    Locate a data file with flexible matching.

    Supports both:
    - {region_id}.{suffix}.csv
    - {region_id}.{suffix}.csv.gz
    If region_id already ends with .{suffix}, assumes user passed full stem.

    Parameters
    ----------
    root : Path
        Directory to search in.
    region_id : str
        Region identifier without suffix, e.g. 'UPMC_c007_v001_r001_reg062'.
    suffix : str
        Type of file: 'cell_data', 'cell_features', 'cell_types', or 'expression'.

    Returns
    -------
    Path
        The matched file path.

    Raises
    ------
    FileNotFoundError
        If no file is found matching the pattern.
    """
    if region_id.endswith(f".{suffix}"):
        base = region_id
    else:
        base = f"{region_id}.{suffix}"

    for ext in [".csv", ".csv.gz"]:
        candidate = root / f"{base}{ext}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No file found for pattern {base}.csv[.gz] in directory: {root}"
    )

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

    # locate each file robustly
    coords_fp   = _find_file(root, region_id, "cell_data")
    features_fp = _find_file(root, region_id, "cell_features")
    types_fp    = _find_file(root, region_id, "cell_types")
    expr_fp     = _find_file(root, region_id, "expression")

    dfs = {
        'coords': pd.read_csv(coords_fp),
        'features': pd.read_csv(features_fp),
        'types': pd.read_csv(types_fp),
        'expr': pd.read_csv(expr_fp),
    }

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
