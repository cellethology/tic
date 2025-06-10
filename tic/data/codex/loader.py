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
from typing import Literal, List, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc

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
    dataset: Literal["upmc", "charville", "dfci"] | str = 'upmc',
    region_id: str = 'UPMC_c001_v001_r001_reg001',
    cache_dir: PathLike = DEFAULT_DATACACHE_DIR,
    *,
    normalize: Literal[None, "size", "counts", "scanpy"] = None,
    log: bool = False,
    preprocessed: bool = False,
    min_total_counts_ratio: Optional[float] = None,
) -> AnnData:
    """
    Load a single spatial region from a CODEX dataset into AnnData.

    Parameters
    ----------
    dataset : str
        CODEX dataset name ('upmc', 'charville', or 'dfci').
    region_id : str
        Region identifier, e.g. 'UPMC_c001_v001_r001_reg001'.
    cache_dir : str or Path
        Root cache directory where datasets are stored.
    normalize : {"size", "counts", "scanpy", None} , will be ignored if preprocessed is True
        Normalization method:
        - "size": normalize by physical cell size
        - "counts": normalize per-cell total to 1.0
        - "scanpy": use `sc.pp.normalize_total(target_sum=1e4)`
        - None: no normalization
    log : bool, will be ignored if preprocessed is True
        Whether to apply `sc.pp.log1p` transformation after normalization.
    preprocessed : bool
        Whether the data is already preprocessed. If True, skip normalization and log1p.
    min_total_counts_ratio : Optional[float]
        Minimum total counts ratio. If provided, filter cells with total counts < mean_count * min_total_counts_ratio.

    Returns
    -------
    AnnData
        - .X: cell-by-biomarker expression matrix (float32)
        - .obs: DataFrame with 'cell_id', 'cell_type', 'size'
        - .var: biomarker names
        - .obsm['spatial']: cell (X, Y) coordinates
        - .uns: metadata including tissue ID
    """
    # Ensure data present
    root = ensure_codex_dataset(dataset, cache_dir)

    # Locate files
    coords_fp   = _find_file(root, region_id, "cell_data")
    features_fp = _find_file(root, region_id, "cell_features")
    types_fp    = _find_file(root, region_id, "cell_types")
    expr_fp     = _find_file(root, region_id, "expression")

    # Read dataframes
    dfs = {
        'coords': pd.read_csv(coords_fp),
        'features': pd.read_csv(features_fp),
        'types': pd.read_csv(types_fp),
        'expr': pd.read_csv(expr_fp),
    }

    # Normalize cell IDs to strings
    for df in dfs.values():
        df['CELL_ID'] = df['CELL_ID'].astype(str).str.strip()

    # Drop acquisition column if present
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

    # Build obs and var
    obs = (
        merged[['CELL_ID', 'CELL_TYPE', 'SIZE']]
        .rename(columns={'CELL_ID': 'cell_id', 'CELL_TYPE': 'cell_type', 'SIZE': 'size'})
        .set_index('cell_id')
    )
    var = pd.DataFrame(index=biom_cols)
    obsm = {'spatial': merged[['X', 'Y']].to_numpy(dtype=np.float32)}

    adata = AnnData(X=X, obs=obs, var=var, obsm=obsm)
    adata.uns['tissue_id'] = region_id
    adata.uns['data_level'] = 'tissue'

        # --- Normalize ---
    if not preprocessed:
        if normalize == 'size':
            if 'size' in adata.obs:
                adata.X = adata.X / adata.obs['size'].values[:, None]
                logger.info("Normalized by cell size.")
            else:
                logger.warning("Requested size normalization but 'size' not found in obs.")
        elif normalize == 'counts':
            totals = np.asarray(adata.X.sum(axis=1)).flatten()
            totals[totals == 0] = 1  # avoid divide-by-zero
            adata.X = adata.X / totals[:, None]
            logger.info("Normalized by total counts (sum = 1).")
        elif normalize == 'scanpy':
            try:
                sc.pp.normalize_total(adata, target_sum=1e4)
                logger.info("Normalized using `sc.pp.normalize_total(target_sum=1e4)`.")
            except Exception as e:
                logger.warning(f"Scanpy normalization failed: {e}")

        # --- Log transform ---
        if log:
            if np.any(adata.X < 0):
                logger.warning("Skipping log1p: expression matrix contains negative values.")
            else:
                adata.X = np.log1p(adata.X)
                logger.info("Applied log1p transformation.")

        # --- Optional filtering by total counts ratio ---
        if min_total_counts_ratio is not None:
            total_counts = adata.X.sum(axis=1)
            mean_count = total_counts.mean()
            threshold = mean_count * min_total_counts_ratio
            keep_mask = total_counts >= threshold
            prev_n = adata.n_obs
            adata = adata[keep_mask].copy()
            logger.info(
                "Filtered cells with total counts < %.2f (%.1fx mean): %d → %d cells",
                threshold,
                min_total_counts_ratio,
                prev_n,
                adata.n_obs,
            )

        # --- Drop cells with NaNs ---
        adata = adata[~np.isnan(adata.X).any(axis=1)]
    else:
        logger.info("Skipping normalization and log1p because `preprocessed=True`.")

    # --- Drop cells with NaNs ---
    adata = adata[~np.isnan(adata.X).any(axis=1)]

    # --- Ensure dense ---
    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()

    # --- Final checks ---
    adata = check_spatial_anndata(adata)
    logger.info("Loaded %s (%s): %d cells * %d biomarkers.",
                dataset, region_id, adata.n_obs, adata.n_vars)
    return adata