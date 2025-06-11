# tic/bgi/loader.py
# ==================

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from ...constant import DEFAULT_DATACACHE_DIR
from ..utils import check_spatial_anndata

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

PathLike = Union[str, Path]

# registry of your available .h5ad samples
BGI_DATASETS: dict[str, str] = {
    "B03425E1": "B03425E1.h5ad",
    "C03628C1": "C03628C1.h5ad",
}


def load_bgi_dataset(
    name: Literal["B03425E1", "C03628C1"],
    data_dir: PathLike = DEFAULT_DATACACHE_DIR,
    *,
    normalize: Literal[None, "counts", "scanpy", "zscore"] = None,
    log: bool = False,
    n_cells: Optional[int] = None,
    min_total_counts_ratio: Optional[float] = None,
) -> AnnData:
    """
    Loader for BGI-GEF–derived AnnData `.h5ad` files.

    Parameters
    ----------
    name
        One of the keys in BGI_DATASETS (e.g. "B03425E1").
    data_dir
        Directory where those .h5ad files live.
    normalize
        If "counts", divide each cell by its total; if "scanpy", use sc.pp.normalize_total;
        if "zscore", per-gene z-score each column.
    log
        If True, apply `np.log1p` after normalization.
    n_cells
        If set and smaller than adata.n_obs, subsample to this many cells
        by closest-to-centroid distance in spatial coords.
    min_total_counts_ratio
        If set, drop any cell with total counts < mean_counts * ratio.
    """
    # 1) find file
    root = Path(data_dir)
    if name not in BGI_DATASETS:
        raise ValueError(f"Unknown sample {name}, available: {list(BGI_DATASETS)}")
    h5ad_path = root / BGI_DATASETS[name]
    if not h5ad_path.exists():
        raise FileNotFoundError(f"BGI .h5ad not found at {h5ad_path}")

    # 2) load
    adata = sc.read_h5ad(h5ad_path)
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str).str.strip()

    if "cell_type" not in adata.obs.columns:
        cache_file = root / f"{name}_llm_cell_types.parquet"
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

    # 3) spatial coords
    if "spatial" in adata.obsm:
        pass
    elif {"x", "y"}.issubset(adata.obs.columns):
        adata.obsm["spatial"] = adata.obs[["x", "y"]].values
    else:
        logger.warning("No spatial coords: missing obsm['spatial'] or obs['x','y'].")

    # 4) densify X
    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()

    # 5) filter low-count cells
    if min_total_counts_ratio is not None:
        total = adata.X.sum(axis=1)
        mean_total = total.mean()
        thresh = mean_total * min_total_counts_ratio
        mask = total >= thresh
        prev = adata.n_obs
        adata = adata[mask].copy()
        logger.info(
            "Filtered cells < %.2f (%.1fx mean): %d → %d",
            thresh, min_total_counts_ratio, prev, adata.n_obs
        )

    # 6) subsample spatially
    if n_cells is not None and n_cells < adata.n_obs:
        coords = adata.obsm.get("spatial")
        if coords is not None:
            center = coords.mean(axis=0)
            d = np.linalg.norm(coords - center, axis=1)
            idx = np.argsort(d)[:n_cells]
            adata = adata[idx].copy()
        else:
            logger.warning("Cannot subsample: no spatial coords.")

    # 7) normalization
    if normalize == "counts":
        sums = adata.X.sum(axis=1)
        adata.X = adata.X / sums[:, None]
        logger.info("Normalized per-cell counts.")
    elif normalize == "scanpy":
        sc.pp.normalize_total(adata, target_sum=1e4)
        logger.info("Normalized with `sc.pp.normalize_total`.")
    elif normalize == "zscore":
        adata.X = (adata.X - adata.X.mean(axis=0)) / adata.X.std(axis=0)
        logger.info("Normalized per-gene z-score.")

    # 8) log transform
    if log:
        adata.X = np.log1p(adata.X)
        logger.info("Applied log1p to X.")

    # 9) finalize
    adata.uns["tissue_id"] = name
    adata.uns["data_level"] = "tissue"
    logger.info("Loaded BGI sample %s: %d × %d", name, adata.n_obs, adata.n_vars)

    check_spatial_anndata(adata)
    return adata