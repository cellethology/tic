"""tic.xenium.loader
====================

Load 10x Genomics Xenium datasets into AnnData objects.

Functions
---------
- load_xenium_dataset: standard Xenium dataset loader
- load_xenium_pancreas_cancer: specialized Pancreas Cancer loader
- _prepare_pancreas_boundaries: extract Zarr-based polygon masks
"""
from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
import zarr
from anndata import AnnData
from tqdm import tqdm

from ...constant import DEFAULT_DATACACHE_DIR
from .download import ensure_xenium_dataset, XENIUM_DATASETS
from ..utils import check_spatial_anndata

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

PathLike = Union[str, Path]


def load_xenium_dataset(
    name: Literal["xenium_ffpe_human_breast", "xenium_kidney_cancer"],
    cache_dir: PathLike = DEFAULT_DATACACHE_DIR,
    *,
    force_download: bool = False,
    normalize: Literal[None, "size", "counts"] = None,
    n_cells: Optional[int] = None,
    include_mask: bool = False,
) -> AnnData:
    """
    Load a standardized Xenium dataset into AnnData.

    Parameters
    ----------
    name : str
        Dataset key from XENIUM_DATASETS, e.g. 'xenium_ffpe_human_breast'.
    cache_dir : Path or str
        Base cache directory.
    force_download : bool
        If True, re-download data even if present.
    normalize : str or None
        'size' (cell_area) or 'counts' (library size) normalization.
    n_cells : int or None
        Subsample to the n most central cells if provided.
    include_mask : bool
        Attach cell/nucleus polygon masks if available.

    Returns
    -------
    AnnData
        .X: expression matrix
        .obs: cell metadata
        .obsm['spatial']: centroids
        .uns: metadata including 'tissue_id','data_level', and optional 'cell_boundaries'
    """
    # Ensure data present
    ds_root = ensure_xenium_dataset(name, cache_dir, force=force_download)
    cfg = XENIUM_DATASETS[name]

    # Expression
    expr_path = ds_root / cfg['expr']
    adata = sc.read_10x_h5(expr_path.as_posix(), gex_only=False)
    adata.var_names_make_unique()

    # Metadata
    cells_df = pd.read_csv(ds_root / cfg['cells'], compression='infer').set_index('cell_id')
    if 'extra' in cfg:
        extra_df = pd.read_csv(ds_root / cfg['extra'])
        cells_df = cells_df.merge(extra_df, on='cell_id', how='left')
        cells_df['cell_type'] = cells_df.get('group', 'Unknown')

    # Align
    adata.obs_names = adata.obs_names.astype(str)
    cells_df.index = cells_df.index.astype(str)
    shared = adata.obs_names.intersection(cells_df.index)
    adata = adata[shared].copy()
    cells_df = cells_df.loc[shared]

    # Attach obs/obsm
    adata.obs = cells_df
    adata.obsm['spatial'] = cells_df[['x_centroid','y_centroid']].values
    if 'cell_type' not in adata.obs:
        adata.obs['cell_type'] = 'unknown'

    # Subsample
    if n_cells is not None and n_cells < adata.n_obs:
        xy = adata.obsm['spatial']
        centre = xy.mean(axis=0)
        d = np.linalg.norm(xy - centre, axis=1)
        keep = np.argsort(d)[:n_cells]
        adata = adata[keep].copy()

    # Normalize
    if normalize == 'size' and 'cell_area' in adata.obs:
        adata.X = adata.X / adata.obs['cell_area'].values[:,None]
    elif normalize == 'counts':
        totals = adata.X.sum(axis=1)
        adata.X = adata.X / totals[:,None]

    # Metadata
    adata.uns.update({'tissue_id': name, 'data_level': 'tissue'})
    logger.info("Loaded %s: %d cells × %d genes.", name, adata.n_obs, adata.n_vars)

    # Masks
    if include_mask:
        adata.uns['cell_boundaries'] = {}
        for mask_type in ['cell','nucleus']:
            for ext in ['.parquet','.csv.gz']:
                path = ds_root / f"{mask_type}_boundaries{ext}"
                if path.exists():
                    df = pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path)
                    df['cell_id'] = df['cell_id'].astype(str)
                    adata.uns['cell_boundaries'][mask_type] = df[df['cell_id'].isin(adata.obs_names)].reset_index(drop=True)
                    logger.info("Loaded %s boundaries from %s", mask_type, path.name)
                    break

    # Ensure dense
    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()
    check_spatial_anndata(adata)
    return adata


def _prepare_pancreas_boundaries(data_root: str) -> None:
    """
    从 cells.zarr.zip 中提取 cell_id、cell 和 nucleus polygon 数据，
    生成并缓存 cell_boundaries.parquet 和 nucleus_boundaries.parquet（如果不存在）。
    """
    cell_parquet = os.path.join(data_root, "cell_boundaries.parquet")
    nuc_parquet  = os.path.join(data_root, "nucleus_boundaries.parquet")

    # 如果两份都已存在，直接跳过
    if os.path.exists(cell_parquet) and os.path.exists(nuc_parquet):
        return

    zarr_zip = os.path.join(data_root, "cells.zarr.zip")
    if not os.path.exists(zarr_zip):
        raise FileNotFoundError(f"[ERROR] 找不到 {zarr_zip}")

    with tempfile.TemporaryDirectory() as tmp:
        # 1) 解压
        with zipfile.ZipFile(zarr_zip, "r") as zf:
            zf.extractall(tmp)
        z = zarr.open(tmp, mode="r")

        # 2) 读取 cell_id 列表
        cid_arr = z["cell_id"][...]  # shape: (N_cells, 2)
        # cell_ids = [f"{int(a)}" for a,b in cid_arr]
        cell_ids = [f"{int(a)}" for a in cid_arr[:,0]]

        # 3) 读取 polygon 顶点和顶点数量
        verts = z["polygon_vertices"]      # (2, N_cells, V_max)
        nv    = z["polygon_num_vertices"]  # (2, N_cells) or (N_cells,)

        # 4) 构造 DataFrame for cell & nucleus
        for idx, (out_path, label) in enumerate([
            (cell_parquet,  "cell"),
            (nuc_parquet,   "nucleus"),
        ]):
            rows = []
            N = len(cell_ids)
            for i in tqdm(range(N), desc=f"Processing {label} polygons"):
                n_v = int(nv[idx, i]) if nv.ndim == 2 else int(nv[i])
                xs = verts[0, i, :n_v]
                ys = verts[1, i, :n_v]
                cid = cell_ids[i]
                for x, y in zip(xs, ys):
                    rows.append({
                        "cell_id": cid,
                        "vertex_x": float(x),
                        "vertex_y": float(y)
                    })
            df = pd.DataFrame(rows)
            df.to_parquet(out_path)
            print(f"[✓] 生成 {label}_boundaries.parquet → {out_path}")

# ── Special Xenium  without standard 10x Genomics zip file ────────────────────────────────────────────────────────────────────

# def load_xenium_pancreas_cancer(
#     data_root: str = os.path.join(DEFAULT_DATACACHE_DIR, "xenium_pancreas_cancer"),
#     tissue_id: str = "Xenium_hPancreas",
#     normalize_method: Optional[Literal["size", "counts"]] = None,
#     n_cells: Optional[int] = None,
#     include_mask: bool = True,
# ) -> AnnData:
#     """
#     Load Xenium human pancreas cancer data with optional normalization and
#     spatial filtering based on automatic circular region selection.

#     Parameters
#     ----------
#     data_root : str
#         Root directory containing Xenium data files.
#     tissue_id : str
#         Identifier for the tissue dataset.
#     normalize_method : Optional[str]
#         Method to normalize expression data. One of 'size', 'counts', or None.
#     n_cells : Optional[int]
#         Number of cells to select from a central spatial region.

#     Returns
#     -------
#     AnnData
#         Annotated data matrix with gene expression, cell metadata, and spatial coordinates.
#     """
#     if not os.path.exists(data_root):
#         ensure_xenium_dataset(name="xenium_pancreas_cancer")

#     feature_matrix = os.path.join(data_root, 'cell_feature_matrix.h5')
#     cells_file = os.path.join(data_root, 'cells.csv.gz')
#     types_file = os.path.join(data_root, 'Xenium_V1_hPancreas_Cancer_Add_on_FFPE_cell_groups.csv')

#     # Read expression and metadata
#     expr_adata = sc.read_10x_h5(feature_matrix)
#     expr_df = pd.DataFrame(
#         expr_adata.X.toarray(),
#         index=expr_adata.obs_names.astype(str),
#         columns=expr_adata.var_names
#     )

#     cells_df = pd.read_csv(cells_file, compression='gzip')
#     types_df = pd.read_csv(types_file)

#     # Ensure cell_id is string
#     cells_df["cell_id"] = cells_df["cell_id"].astype(str)
#     types_df["cell_id"] = types_df["cell_id"].astype(str)

#     # Merge cell type info into cells
#     merged = cells_df.merge(types_df, how="left", on="cell_id").set_index("cell_id")
#     merged = merged.loc[expr_df.index]  # Align with expression data
#     merged["cell_type"] = merged.get("group", "Unknown")

#     # Spatial filtering by adaptive circular region
#     if n_cells is not None:
#         coords = merged[["x_centroid", "y_centroid"]].copy()

#         # Step 1: Compute geometric center of the tissue
#         center_x = coords["x_centroid"].mean()
#         center_y = coords["y_centroid"].mean()

#         # Step 2: Compute distance to center for each cell
#         coords["dist_to_center"] = np.sqrt(
#             (coords["x_centroid"] - center_x) ** 2 +
#             (coords["y_centroid"] - center_y) ** 2
#         )

#         # Step 3: Select n closest cells
#         selected = coords.sort_values("dist_to_center").iloc[:n_cells]
#         selected_ids = selected.index

#         # Step 4: Use farthest distance among selected cells as effective radius
#         final_radius = selected["dist_to_center"].max()

#         expr_df = expr_df.loc[selected_ids]
#         merged = merged.loc[selected_ids]

#         print(
#             f"[INFO] Selected {n_cells} cells within radius "
#             f"{final_radius:.2f} from center ({center_x:.2f}, {center_y:.2f})"
#         )

#     # Construct .obs
#     obs = merged[["cell_type"]].copy()
#     obs.index.name = "cell_id"
#     obs["cell_area"] = merged.get("cell_area")
#     obs["total_counts"] = expr_df.sum(axis=1)

#     # Normalize if requested
#     X = expr_df.loc[obs.index].values.astype(float)

#     if normalize_method == "size" and obs["cell_area"].notna().all():
#         X = X / obs["cell_area"].values[:, None]
#     elif normalize_method == "counts":
#         X = X / X.sum(axis=1, keepdims=True)

#     # Build AnnData object
#     adata = AnnData(
#         X=X,
#         obs=obs,
#         var=pd.DataFrame(index=expr_df.columns),
#         obsm={"spatial": merged.loc[obs.index, ["x_centroid", "y_centroid"]].to_numpy()}
#     )
#     adata.uns["tissue_id"] = tissue_id
#     adata.uns["data_level"] = "tissue"

#     if include_mask:
#         adata.uns['cell_boundaries'] = {}
#         for mask_type in ['cell', 'nucleus']:
#             for ext in ['.parquet', '.csv.gz']:
#                 path = Path(data_root) / f"{mask_type}_boundaries{ext}"
#                 if path.exists():
#                     df = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
#                     df['cell_id'] = df['cell_id'].astype(str)
#                     adata.uns['cell_boundaries'][mask_type] = df[df['cell_id'].isin(adata.obs_names)].reset_index(drop=True)
#                     logger.info("Loaded %s boundaries from %s", mask_type, path.name)
#                     break

#     # Ensure dense
#     if not isinstance(adata.X, np.ndarray):
#         adata.X = adata.X.toarray()

#     print(f"Successfully loaded {tissue_id} data: {adata}")
#     return adata


# TODO: Use auto-downloaded pancreas cancer data to load as anndata object, not use jerry's data