# tic/bgi/loader.py
# ==================
"""BGI-GEF dataset loader **with automatic LLM cell-type annotation**
--------------------------------------------------------------------

This module exposes :func:`load_bgi_dataset`, a high-level helper that

* loads a converted BGI `.h5ad` file (raw / tissue level);
* ensures `obsm["spatial"]` exists;
* supports common filtering, subsampling, normalisation & log1p;
* **runs LLM-powered cell-type annotation when missing**, caching the result to
  ``{sample_id}_{data_type}_llm_cell_types.parquet`` so future calls are
  instantaneous;
* crucially **keeps the full gene matrix intact** – highly-variable-gene (HVG)
  sub-setting is performed on a *copy* of the data used solely for clustering
  + annotation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from ...constant import DEFAULT_DATACACHE_DIR
from ..utils import check_spatial_anndata
from ...annotation import annotate_adata  # noqa: E402 – local import OK here

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

PathLike = Union[str, Path]

# -----------------------------------------------------------------------------
# Sample descriptions – passed to LLM for extra context (extend as needed)
# -----------------------------------------------------------------------------
_DESCRIPTIONS: Dict[str, str] = {
    "B03425E1": "Non-small-cell lung cancer (NSCLC).",
    "C03628C1": "Clear-cell renal cell carcinoma.",
    "B02804B5": "Murine MC38 orthotopic colorectal cancer (CRC).",
    "B03425E3": "Breast cancer sample.",
}

# -----------------------------------------------------------------------------
# Loader
# -----------------------------------------------------------------------------

def load_bgi_dataset(
    sample_id: str,
    data_type: Literal["raw", "tissue", "cellbin"] = "tissue",
    data_dir: PathLike = DEFAULT_DATACACHE_DIR,
    *,
    normalize: Literal[None, "counts", "scanpy", "zscore"] = None,
    log: bool = False,
    n_cells: Optional[int] = None,
    min_total_counts_ratio: Optional[float] = None,
    auto_annotate: bool = True,
    force_reannotate: bool = False,
    assign_params: Optional[dict] = None,
    llm_model_name: Optional[str] = None,
    llm_kwargs: Optional[dict] = None,
    cache_parquet_kwargs: Optional[dict] = None,
) -> AnnData:
    """Load a BGI sample and (optionally) annotate cell types via LLM.

    Parameters are identical to the previous revision; functional changes are:

    * run **annotation on a copy** of the input so *original genes stay put*;
    * accept either ``cell_type`` or ``pred_cell_type`` column names returned
      by :func:`annotate_adata` and rename consistently;
    * always copy back **clustering labels** (``leiden``) so they remain
      available on the full-gene `AnnData`.
    """

    # ------------------------------------------------------------------
    # Locate & read
    # ------------------------------------------------------------------
    sample_dir = Path(data_dir) / sample_id
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory {sample_dir} not found.")

    h5ad_file = sample_dir / f"{sample_id}.{data_type}.h5ad"
    if not h5ad_file.exists():
        raise FileNotFoundError(f"{data_type} file not found: {h5ad_file}")

    adata = sc.read_h5ad(h5ad_file)
    adata.var_names_make_unique()
    adata.obs_names = adata.obs_names.astype(str).str.strip()

    # ------------------------------------------------------------------
    # Spatial coords
    # ------------------------------------------------------------------
    if "spatial" not in adata.obsm and {"x", "y"}.issubset(adata.obs.columns):
        adata.obsm["spatial"] = adata.obs[["x", "y"]].values
    if "spatial" not in adata.obsm:
        logger.warning("No spatial coords for %s – downstream spatial ops may fail.", sample_id)

    # ------------------------------------------------------------------
    # Filter / subsample / normalise
    # ------------------------------------------------------------------
    if min_total_counts_ratio is not None:
        totals = adata.X.sum(axis=1)
        thresh = totals.mean() * min_total_counts_ratio
        adata = adata[totals >= thresh].copy()
        logger.info("Filtered cells below %.1fx mean total counts (≥ %.0f).", min_total_counts_ratio, thresh)

    if n_cells is not None and n_cells < adata.n_obs:
        coords = adata.obsm.get("spatial")
        if coords is None:
            logger.warning("Requested subsample but spatial coords missing – skipping.")
        else:
            centre = coords.mean(axis=0)
            keep_idx = np.linalg.norm(coords - centre, axis=1).argsort()[:n_cells]
            adata = adata[keep_idx].copy()

    if normalize == "counts":
        adata.X = adata.X / adata.X.sum(axis=1, keepdims=True)
    elif normalize == "scanpy":
        sc.pp.normalize_total(adata, target_sum=1e4)
    elif normalize == "zscore":
        adata.X = (adata.X - adata.X.mean(0)) / adata.X.std(0)

    if log:
        adata.X = np.log1p(adata.X)

    # ------------------------------------------------------------------
    # Cell-type annotation – try cache, else LLM
    # ------------------------------------------------------------------
    cache_file = sample_dir / f"{sample_id}_{data_type}_llm_cell_types.parquet"
    if "cell_type" not in adata.obs.columns:
        # 1. Try cached parquet first ------------------------------------------------
        if cache_file.exists() and not force_reannotate:
            try:
                df_cache = pd.read_parquet(cache_file).set_index("cell_id")
                adata.obs["cell_type"] = (
                    df_cache.loc[adata.obs_names, "pred_cell_type"].astype("category")
                )
                logger.info("Loaded cached cell types from %s", cache_file)
            except Exception as exc:
                logger.warning("Failed reading cache %s – %s", cache_file, exc)

        # 2. Run LLM annotation if allowed ------------------------------------------
        elif auto_annotate or force_reannotate:
            if annotate_adata is None:
                logger.warning("annotate_adata unavailable – cannot auto-annotate.")
            else:
                assign_params = assign_params or {
                    "min_cells": 0,
                    "min_genes": 0,
                    "n_pca": 50,
                    "n_neighbors": 15,
                    "resolution": 2.0,
                    "n_filter_highly_variable_genes": 2000,
                }
                llm_model_name = llm_model_name or "openai"
                llm_kwargs = llm_kwargs or {}
                desc = _DESCRIPTIONS.get(sample_id, "")

                logger.info("Running LLM annotation for %s (%s) via %s …", sample_id, data_type, llm_model_name)

                # -------- Work on a *copy* so gene set reductions don't leak -------
                work = adata.copy()
                work, cluster_map = annotate_adata(
                    adata=work,
                    assign_params=assign_params,
                    llm_model_name=llm_model_name,
                    llm_kwargs=llm_kwargs,
                    dataset_description=desc,
                    groupby="leiden",
                    rank_genes_groups_method="t-test",
                    n_genes=10,
                    copy=False,
                    return_cluster_annotation=True,
                )

                # ---- bring back labels & clustering into the *full* AnnData -------
                for key in ("leiden", "cell_type", "pred_cell_type"):
                    if key in work.obs.columns:
                        adata.obs[key] = work.obs[key]

                if "cell_type" not in adata.obs.columns and "pred_cell_type" in adata.obs.columns:
                    adata.obs["cell_type"] = adata.obs["pred_cell_type"].astype("category")

                # Last resort: map via cluster_map
                if "cell_type" not in adata.obs.columns and cluster_map:
                    adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_map).astype("category")

                # ------------------------- cache to parquet -----------------------
                if "cell_type" in adata.obs.columns:
                    df_cache = (
                        adata.obs[["cell_type"]]
                        .rename(columns={"cell_type": "pred_cell_type"})
                        .reset_index()
                        .rename(columns={adata.obs.index.name or "index": "cell_id"})
                    )
                    try:
                        df_cache.to_parquet(cache_file, **(cache_parquet_kwargs or {}))
                        logger.info("Saved annotation cache → %s", cache_file)
                    except Exception as exc:
                        logger.warning("Could not write cache %s – %s", cache_file, exc)
                else:
                    logger.warning("LLM annotation ran but no 'cell_type' inferred – skipping cache save.")

    # ------------------------------------------------------------------
    # House-keeping & return
    # ------------------------------------------------------------------
    adata.uns["tissue_id"] = sample_id
    adata.uns["data_level"] = data_type

    try:
        check_spatial_anndata(adata)
    except Exception as exc:
        logger.warning("Spatial integrity check failed: %s", exc)

    logger.info("Loaded BGI sample %s (%s) – %d cells × %d genes", sample_id, data_type, adata.n_obs, adata.n_vars)
    return adata
