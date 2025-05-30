# tic/codex/loader.py
"""
Load pre-downloaded Murine CODEX datasets into `anndata.AnnData` objects.

Functions
---------
- list_regions: Enumerate spatial region identifiers in a CODEX dataset.
- load_region: Load expression and metadata for a single CODEX region, with optional normalization.
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

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

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
    *,
    normalize: Literal[None, "size", "counts"] = None,
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
    normalize
        If 'size', divides each cell's counts by its recorded size.
        If 'counts', divides by total counts per cell.
        If None, no normalization is applied.

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
    # Ensure data present
    root = ensure_codex_dataset(dataset, cache_dir)

    # Locate files
    coords_fp   = _find_file(root, region_id, "cell_data")
    features_fp = _find_file(root, region_id, "cell_features")
    types_fp    = _find_file(root, region_id, "cell_types")
    expr_fp     = _find_file(root, region_id, "expression")

    # Read
    dfs = {
        'coords': pd.read_csv(coords_fp),
        'features': pd.read_csv(features_fp),
        'types': pd.read_csv(types_fp),
        'expr': pd.read_csv(expr_fp),
    }

    # Normalize cell IDs to strings
    for df in dfs.values():
        df['CELL_ID'] = df['CELL_ID'].astype(str).str.strip()

    # Drop acquisition ID if present
    expr = dfs['expr']
    if 'ACQUISITION_ID' in expr.columns:
        expr = expr.drop(columns=['ACQUISITION_ID'])

    # Merge all tables
    merged = (
        dfs['coords']
        .merge(dfs['features'], on='CELL_ID')
        .merge(dfs['types'], on='CELL_ID')
        .merge(expr, on='CELL_ID')
    )

    # Build expression matrix
    biom_cols = [c for c in expr.columns if c != 'CELL_ID']
    X = merged[biom_cols].to_numpy(dtype=np.float32)

    # Build obs, var, obsm
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

    # --- Normalize ---
    if normalize == 'size':
        if 'size' in adata.obs:
            adata.X = adata.X / adata.obs['size'].values[:, None]
        else:
            logger.warning("Requested size normalization but 'size' not in obs.")
    elif normalize == 'counts':
        totals = np.asarray(adata.X.sum(axis=1)).flatten()
        adata.X = adata.X / totals[:, None]

    # drop Nans
    adata = adata[~np.isnan(adata.X).any(axis=1)]

    # Final checks
    adata = check_spatial_anndata(adata)
    logger.info("Loaded %s (%s): %d cells * %d biomarkers.",
                dataset, region_id, adata.n_obs, adata.n_vars)
    return adata