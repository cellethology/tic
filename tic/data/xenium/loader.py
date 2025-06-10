# tic/xenium/loader.py
# ====================

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

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
    normalize: Literal[None, "size", "counts", "scanpy", "zscore"] = None,
    log: bool = False,
    n_cells: Optional[int] = None,
    include_mask: bool = False,
    min_total_counts_ratio: Optional[float] = None,
) -> AnnData:
    """
    Robust loader for Xenium datasets into AnnData, with support for expression normalization
    and log transformation.  Additionally, if the original dataset lacks a 'cell_type'
    column but a cache file named '{name}_llm_cell_types.parquet' already exists in the
    same cache folder, that Parquet will be loaded into adata.obs['cell_type'].

    Parameters
    ----------
    name : str
        Xenium dataset name.
    cache_dir : str or Path
        Directory for cached data.
    force_download : bool
        Force re-download of dataset.
    normalize : {"size", "counts", "scanpy", "zscore", None}
        Type of normalization.
    log : bool
        Whether to apply `sc.pp.log1p` after normalization.
    n_cells : int or None
        Subsample to fixed number of cells.
    include_mask : bool
        Whether to load segmentation boundaries.
    min_total_counts_ratio : Optional[float]
        Minimum total counts ratio. If provided, filter cells with total counts < mean_count * min_total_counts_ratio.
    """
    # 1) Ensure data present on disk
    ds_root = Path(ensure_xenium_dataset(name, cache_dir, force=force_download))
    cfg = XENIUM_DATASETS[name]

    # 2) Load raw expression matrix
    expr_path = ds_root / cfg["expr"]
    adata = sc.read_10x_h5(expr_path.as_posix(), gex_only=False)
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str).str.strip()

    # 3) Load or build cells metadata
    cells_path = _choose_file(ds_root, "cells", [".parquet", ".csv.gz"])
    if cells_path is None:
        raise FileNotFoundError(f"No cells file found in {ds_root}")
    if cells_path.suffix == ".parquet":
        cells_df = pd.read_parquet(cells_path)
    else:
        cells_df = pd.read_csv(cells_path, compression="infer")
    cells_df["cell_id"] = cells_df["cell_id"].astype(str).str.strip()

    # 4) Merge in any “extra” metadata (e.g., 'group' => 'cell_type')
    if "extra" in cfg:
        extra_path = Path(cfg["extra"])
        extra_df = pd.read_csv(ds_root / extra_path)
        extra_df["cell_id"] = extra_df["cell_id"].astype(str).str.strip()
        cells_df = cells_df.merge(extra_df, on="cell_id", how="left")
        if "group" in cells_df.columns:
            cells_df["cell_type"] = cells_df["group"].fillna("Unknown")

    # 5) Reindex by cell_id and align with adata.obs_names
    cells_df = cells_df.set_index("cell_id")
    shared = adata.obs_names.intersection(cells_df.index)
    if len(shared) == 0:
        logger.error(
            "No shared cell IDs between expression and metadata.\n"
            f"adata.obs_names sample: {adata.obs_names[:5]}\n"
            f"cells_df.index sample: {cells_df.index[:5]}"
        )
        raise ValueError("Alignment failed: no overlapping cell IDs.")
    adata = adata[shared].copy()
    cells_df = cells_df.loc[shared]
    adata.obs = cells_df

    # 6) If original metadata already provided 'cell_type', keep it.
    #    Otherwise, look for an LLM‐annotated cache file.
    if "cell_type" not in adata.obs.columns:
        cache_file = ds_root / f"{name}_llm_cell_types.parquet"
        if cache_file.exists():
            try:
                df_cache = pd.read_parquet(cache_file)
                # Expect df_cache to have columns ["cell_id", "cell_type"]
                df_cache = df_cache.set_index("cell_id")
                df_cache = df_cache.loc[adata.obs_names.intersection(df_cache.index), :]
                adata.obs["cell_type"] = df_cache["pred_cell_type"].astype("category")
                logger.info("Loaded LLM‐annotated cell types from %s", cache_file)
            except Exception as e:
                logger.warning(
                    "Failed to read LLM cache (%s): %s. Skipping cell_type load.", cache_file, e
                )

    # 7) Spatial coordinates
    if {"x_centroid", "y_centroid"}.issubset(adata.obs.columns):
        adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].values
    else:
        logger.warning("Centroid columns missing in cells metadata.")

    # 8) Optionally subsample to n_cells
    if n_cells is not None and n_cells < adata.n_obs:
        xy = adata.obsm.get("spatial", None)
        if xy is not None:
            centre = xy.mean(axis=0)
            d = np.linalg.norm(xy - centre, axis=1)
            keep = np.argsort(d)[:n_cells]
            adata = adata[keep].copy()
        else:
            logger.warning("Cannot subsample by distance: no spatial coords found.")

    # 9) Ensure dense array
    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()

    # 9.5) Optional filter by total counts ratio
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

    # 10) Normalization
    if normalize == "size" and "cell_area" in adata.obs:
        adata.X = adata.X / adata.obs["cell_area"].values[:, None]
        logger.info("Normalized by cell area.")
    elif normalize == "counts":
        totals = np.array(adata.X.sum(axis=1)).flatten()
        adata.X = adata.X / totals[:, None]
        logger.info("Normalized by total expression (sum = 1).")
    elif normalize == "scanpy":
        sc.pp.normalize_total(adata, target_sum=1e4)
        logger.info("Normalized using `sc.pp.normalize_total(target_sum=1e4)`.")
    elif normalize == "zscore":
        adata.X = (adata.X - adata.X.mean(axis=0)) / adata.X.std(axis=0)
        logger.info("Normalized using `zscore`.")

    # 11) Log transform
    if log:
        adata.X = np.log1p(adata.X)
        logger.info("Applied log1p transformation.")

    # 12) Load boundaries if requested
    if include_mask:
        adata.uns["cell_boundaries"] = {}
        for mask in ["cell", "nucleus"]:
            df_mask = None
            mask_path = _choose_file(ds_root, f"{mask}_boundaries", [".parquet", ".csv.gz"])
            if mask_path is not None:
                if mask_path.suffix == ".parquet":
                    df_mask = pd.read_parquet(mask_path)
                else:
                    df_mask = pd.read_csv(mask_path)
                df_mask["cell_id"] = df_mask["cell_id"].astype(str).str.strip()
                df_mask = df_mask[df_mask["cell_id"].isin(adata.obs_names)].reset_index(drop=True)
                adata.uns["cell_boundaries"][mask] = df_mask
                logger.info("Loaded %s boundaries: %s", mask, mask_path.name)

    # 13) Finalize
    adata.uns.update({"tissue_id": name, "data_level": "tissue"})
    logger.info("Loaded %s: %d cells * %d genes.", name, adata.n_obs, adata.n_vars)

    check_spatial_anndata(adata)
    return adata