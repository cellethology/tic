"""tic.xenium.loader
====================

Load 10x Genomics Xenium datasets into AnnData objects.

Functions
---------
- load_xenium_dataset: standard Xenium dataset loader
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc

from ...constant import DEFAULT_DATACACHE_DIR
from .download import ensure_xenium_dataset, XENIUM_DATASETS
from ..utils import check_spatial_anndata

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

PathLike = Union[str, Path]


def _choose_file(root: Path, base: str, exts: list[str]) -> Optional[Path]:
    """
    Return the first existing file for base + ext under root, or None.
    """
    for ext in exts:
        p = root / f"{base}{ext}"
        if p.exists():
            return p
    return None


def load_xenium_dataset(
    name: Literal[
        "xenium_ffpe_human_breast",
        "xenium_kidney_cancer",
        "xenium_pancreas_cancer",
        "xenium_invasive_lung_cancer",
    ],
    cache_dir: PathLike = DEFAULT_DATACACHE_DIR,
    *,
    force_download: bool = False,
    normalize: Literal[None, "size", "counts"] = None,
    n_cells: Optional[int] = None,
    include_mask: bool = False,
) -> AnnData:
    """
    Robust loader for Xenium datasets into AnnData, auto-detecting file formats
    and ensuring consistent string-based cell IDs across all files.
    """
    # Ensure data present
    ds_root = ensure_xenium_dataset(name, cache_dir, force=force_download)
    cfg = XENIUM_DATASETS[name]

    # --- Expression ---
    expr_path = ds_root / cfg['expr']
    adata = sc.read_10x_h5(expr_path.as_posix(), gex_only=False)
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str).str.strip()

    # --- Metadata: cells ---
    cells_path = _choose_file(ds_root, 'cells', ['.parquet', '.csv.gz'])
    if cells_path is None:
        raise FileNotFoundError(f"No cells file found in {ds_root}")
    if cells_path.suffix == '.parquet':
        cells_df = pd.read_parquet(cells_path)
    else:
        cells_df = pd.read_csv(cells_path, compression='infer')
    cells_df['cell_id'] = cells_df['cell_id'].astype(str).str.strip()

    # Extra metadata
    if 'extra' in cfg:
        extra_path = Path(cfg['extra'])
        extra_df = pd.read_csv(ds_root / extra_path)
        extra_df['cell_id'] = extra_df['cell_id'].astype(str).str.strip()
        cells_df = cells_df.merge(extra_df, on='cell_id', how='left')
        if 'group' in cells_df:
            cells_df['cell_type'] = cells_df['group'].fillna('Unknown')

    # Reindex
    cells_df = cells_df.set_index('cell_id')

    # --- Align ---
    shared = adata.obs_names.intersection(cells_df.index)
    if len(shared) == 0:
        logger.error("No shared cell IDs between expression and metadata.\n"
                     f"adata.obs_names sample: {adata.obs_names[:5]}\n"
                     f"cells_df.index sample: {cells_df.index[:5]}")
        raise ValueError("Alignment failed: no overlapping cell IDs.")
    adata = adata[shared].copy()
    cells_df = cells_df.loc[shared]
    adata.obs = cells_df

    # --- Spatial coords ---
    if {'x_centroid', 'y_centroid'}.issubset(adata.obs.columns):
        adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].values
    else:
        logger.warning("Centroid columns missing in cells metadata.")

    # --- Subsample ---
    if n_cells is not None and n_cells < adata.n_obs:
        xy = adata.obsm['spatial']
        centre = xy.mean(axis=0)
        d = np.linalg.norm(xy - centre, axis=1)
        keep = np.argsort(d)[:n_cells]
        adata = adata[keep].copy()

    # --- Normalize ---
    if normalize == 'size' and 'cell_area' in adata.obs:
        adata.X = adata.X / adata.obs['cell_area'].values[:, None]
    elif normalize == 'counts':
        totals = np.array(adata.X.sum(axis=1)).flatten()
        adata.X = adata.X / totals[:, None]

    # --- Masks (boundaries) ---
    if include_mask:
        adata.uns['cell_boundaries'] = {}
        for mask in ['cell', 'nucleus']:
            df_mask = None
            mask_path = _choose_file(ds_root, f"{mask}_boundaries", ['.parquet', '.csv.gz'])
            if mask_path is not None:
                if mask_path.suffix == '.parquet':
                    df_mask = pd.read_parquet(mask_path)
                else:
                    df_mask = pd.read_csv(mask_path)
                df_mask['cell_id'] = df_mask['cell_id'].astype(str).str.strip()
                df_mask = df_mask[df_mask['cell_id'].isin(adata.obs_names)].reset_index(drop=True)
                adata.uns['cell_boundaries'][mask] = df_mask
                logger.info("Loaded %s boundaries: %s", mask, mask_path.name)

    # --- Finalize ---
    adata.uns.update({'tissue_id': name, 'data_level': 'tissue'})
    logger.info("Loaded %s: %d cells * %d genes.", name, adata.n_obs, adata.n_vars)
    
    # Ensure dense matrix
    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()
    check_spatial_anndata(adata)
    return adata