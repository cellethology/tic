# tic.data.loader
"""
This module provides functions to load and process various spatial transcriptomics datasets.

It includes:
- Loading Codex datasets: Codex-UPMC, Codex-Charville, Codex-DFCI
- Loading Xenium datasets: Xenium-Pancreas-Cancer, Xenium-Colorectal-Cancer
"""
import os
from typing import Literal, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from ..constant import DEFAULT_DATACACHE_DIR
from .io import XENIUM_DATASETS, download_codex_dataset, download_xenium_dataset, download_xenium_pancreas_cancer_data, download_xenium_colorectal_cancer_data

def list_codex_datasets(
    dataset: str | Literal["upmc", "charville", "dfci"] = "upmc",
    dataset_root: str | None = None,
    ) -> list[str]:
    """
    List the available regions in the given Codex dataset.

    Parameters
    ----------
    dataset: str | Literal["upmc", "charville", "dfci"] = "upmc"
        The dataset to list.
    dataset_root: str | None = None
        The root directory to store the dataset. If None, the dataset will be downloaded to the default cache directory. 
    
    Returns
    -------
    list[str]
        The list of available regions in the given Codex dataset.

    Examples
    --------
    >>> list_codex_datasets()
    ['UPMC_c001_v001_r001_reg001', 'UPMC_c001_v001_r001_reg002']
    >>> list_codex_datasets(dataset="charville")
    ['Charville_c001_v001_r001_reg001', 'Charville_c001_v001_r001_reg002']
    >>> list_codex_datasets(dataset="dfci", dataset_root="/path/to/dfci")
    ['s271_c001_v001_r001_reg001', 's271_c001_v001_r001_reg004', 's271_c001_v001_r001_reg004']
    """
    mapping = {
        "upmc": "codex_upmc",
        "charville": "codex_charville",
        "dfci": "codex_dfci",
    }
    dataset_root = os.path.join(DEFAULT_DATACACHE_DIR, mapping[dataset]) if dataset_root is None else dataset_root

    if not os.path.exists(dataset_root):
        print(f"[INFO] Downloading Codex {dataset.capitalize()} data to {dataset_root}")
        download_codex_dataset(dataset=dataset)
    
    def get_region_ids(dir_path):
        """Get all region IDs based on JSON files in the directory."""
        region_ids = []
        for file in os.listdir(dir_path):
            if file.endswith(".cell_data.csv"):
                region_ids.append(file.split(".")[0])
        return sorted(region_ids)

    return get_region_ids(dataset_root)
    



def load_codex_dataset(
    dataset: str | Literal["upmc", "charville", "dfci"] = "upmc",
    dataset_root: str | None = None,
    region_id: str = "UPMC_c001_v001_r001_reg001",
) -> AnnData:
    """
    Load Codex dataset
    
    Parameters
    ----------
    dataset: str | Literal["upmc", "charville", "dfci"] = "upmc"
        The dataset to load.
    dataset_root: str | None = None
        The root directory to store the dataset. If None, the dataset will be downloaded to the default cache directory. 
        Else, the dataset will be loaded from the given directory.
    region_id: str = "UPMC_c001_v001_r001_reg001"
        The region id to load.

    Returns
    -------
    AnnData
        .obs:
            - cell_id: the cell id
            - cell_type: the cell type
            - size: the size of the cell
        .var:
            - biomarker_cols: the biomarker columns
        .obsm:
            - spatial: the spatial coordinates
        .uns:
            - tissue_id: the tissue id
            - data_level: the data level
    """
    mapping = {
        "upmc": "codex_upmc",
        "charville": "codex_charville",
        "dfci": "codex_dfci",
    }
    dataset_root = os.path.join(DEFAULT_DATACACHE_DIR, mapping[dataset]) if dataset_root is None else dataset_root

    if not os.path.exists(dataset_root):
        print(f"[INFO] Downloading Codex {dataset.capitalize()} data to {dataset_root}")
        download_codex_dataset(dataset=dataset)
    
    file_template = {
        "coords": "{region_id}.cell_data.csv",
        "features": "{region_id}.cell_features.csv",
        "types": "{region_id}.cell_types.csv",
        "expression": "{region_id}.expression.csv"
    }
    paths = {key: os.path.join(dataset_root, tpl.format(region_id=region_id)) for key, tpl in file_template.items()}
    dfs = {key: pd.read_csv(path) for key, path in paths.items()}

    for df in dfs.values():
        df["CELL_ID"] = df["CELL_ID"].astype(str)

    if "ACQUISITION_ID" in dfs["expression"].columns:
        dfs["expression"].drop(columns=["ACQUISITION_ID"], inplace=True)

    merged = dfs["coords"].merge(dfs["features"], on="CELL_ID") \
                          .merge(dfs["types"], on="CELL_ID") \
                          .merge(dfs["expression"], on="CELL_ID")

    biomarker_cols = [col for col in dfs["expression"].columns if col != "CELL_ID"]
    X = merged[biomarker_cols].to_numpy()

    obs = merged[["CELL_ID", "CELL_TYPE", "SIZE"]].rename(columns={
        "CELL_ID": "cell_id",
        "CELL_TYPE": "cell_type",
        "SIZE": "size"
    })
    var = pd.DataFrame(index=biomarker_cols)
    obsm = {"spatial": merged[["X", "Y"]].to_numpy()}

    adata = AnnData(X=X, obs=obs, var=var, obsm=obsm)
    adata.uns["tissue_id"] = region_id
    adata.uns["data_level"] = "tissue"
    return adata

# ── Xenium ────────────────────────────────────────────────────────────────────
def load_xenium_dataset(
    dataset: str | Literal['xenium_ffpe_human_breast'],
    cache_dir: str = DEFAULT_DATACACHE_DIR,
    *,
    force_download: bool = False,
    normalize: Literal[None, "size", "counts"] = None,
    n_cells: int | None = None,
    include_mask: bool | None = None,
) -> AnnData:
    """
    Load Standardized AnnData from Xenium dataset. The raw data should at least contain:
    - cell_feature_matrix.h5
    - cells.csv.gz
    - (optional) cell_groups.csv

    Parameters
    ----------
    dataset: str
        The dataset to load.
    cache_dir: str = DEFAULT_DATACACHE_DIR
        The directory to store the downloaded dataset.
    force_download: bool = False
        If True, the dataset will be downloaded even if it already exists.
    normalize: Literal[None, "size", "counts"] = None
        If "size", normalize the expression matrix by cell area.
        If "counts", normalize the expression matrix by total counts.
    n_cells: int | None = None
        If not None, sample n_cells from the central spatial region.
    include_mask: bool | None = None
        If not None, include the mask in the AnnData.uns

    Returns  
    -------
    AnnData
        .X: the expression matrix
        .obs: the cell table
        .obsm: the spatial coordinates
            - spatial: the spatial coordinates
            - data_level: 'tissue'
            - cell_boundaries: the cell boundaries
                - 'cell': the cell boundaries dataframe, column: 'cell_id', 'vertex_x', 'vertex_y'
                - 'nucleus': the nucleus boundaries dataframe, column: 'cell_id', 'vertex_x', 'vertex_y'
        .uns: the metadata
            - tissue_id: the tissue id
            - data_level: the data level
    """
    ds_dir = download_xenium_dataset(dataset, cache_dir, force=force_download)
    cfg = XENIUM_DATASETS[dataset]

    # 1) expression matrix
    expr_path = os.path.join(ds_dir, cfg["expr"])
    adata = sc.read_10x_h5(expr_path, gex_only=False)
    adata.var_names_make_unique()

    # 2) cell table
    cells_path = os.path.join(ds_dir, cfg["cells"])
    cells_df = pd.read_csv(cells_path, compression="infer").set_index("cell_id")

    # (optional) cell type additional table
    if "extra" in cfg:
        extra_path = os.path.join(ds_dir, cfg["extra"])
        if os.path.exists(extra_path):
            types_df = pd.read_csv(extra_path)
            cells_df = cells_df.merge(types_df, how="left", on="cell_id")
            cells_df["cell_type"] = cells_df.get("group", "Unknown")

    # 3) align
    adata.obs_names = adata.obs_names.astype(str)
    cells_df.index = cells_df.index.astype(str)
    shared = adata.obs_names.intersection(cells_df.index)
    adata = adata[shared].copy()
    cells_df = cells_df.loc[shared]

    # 4) obs / obsm
    adata.obs = cells_df
    adata.obsm["spatial"] = cells_df[["x_centroid", "y_centroid"]].values
    if "cell_type" not in adata.obs:
        adata.obs["cell_type"] = "unknown"

    # 5) (optional) circular n_cells sampling (same as pancreas logic)
    if n_cells is not None:
        xy = adata.obsm["spatial"]
        ctr = xy.mean(axis=0)
        d = np.linalg.norm(xy - ctr, axis=1)
        keep = np.argsort(d)[:n_cells]
        adata = adata[keep].copy()

    # 6) (optional) normalize
    if normalize == "size" and "cell_area" in adata.obs:
        adata.X = adata.X / adata.obs["cell_area"].values[:, None]
    elif normalize == "counts":
        totals = adata.X.sum(axis=1)
        adata.X = adata.X / totals[:, None]

    # 7) Metadata
    adata.uns["tissue_id"] = dataset
    adata.uns["data_level"] = "tissue"
    print(f"[✓] Loaded {dataset}: {adata.n_obs} cells × {adata.n_vars} genes.")

    # 8) (optional) load cell/nucleus mask polygon data
    if include_mask:
        boundary_files = {
            "cell": ["cell_boundaries.parquet", "cell_boundaries.csv.gz"],
            "nucleus": ["nucleus_boundaries.parquet", "nucleus_boundaries.csv.gz"]
        }
        # make sure the file exists
        for mask_type, filenames in boundary_files.items():
            for fname in filenames:
                path = os.path.join(ds_dir, fname)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File {path} not found")
        
        for mask_type, filenames in boundary_files.items():
            for fname in filenames:
                path = os.path.join(ds_dir, fname)
                if os.path.exists(path):
                    print(f"[INFO] Loading {mask_type} boundaries from: {fname}")
                    df = (
                        pd.read_parquet(path)
                        if path.endswith(".parquet")
                        else pd.read_csv(path)
                    )
                    # Ensure 'cell_id' is str for join consistency
                    df["cell_id"] = df["cell_id"].astype(str)
                    if "cell_boundaries" not in adata.uns:
                        adata.uns["cell_boundaries"] = {}
                    adata.uns["cell_boundaries"][mask_type] = df
                    break  # Stop once one format is loaded
    return adata


# ── Special Xenium  without standard 10x Genomics zip file ────────────────────────────────────────────────────────────────────

def load_xenium_pancreas_cancer(
    data_root: str = os.path.join(DEFAULT_DATACACHE_DIR, "xenium_pancreas_cancer"),
    tissue_id: str = "Xenium_hPancreas",
    normalize_method: Optional[Literal["size", "counts"]] = None,
    n_cells: Optional[int] = None,
) -> AnnData:
    """
    Load Xenium human pancreas cancer data with optional normalization and
    spatial filtering based on automatic circular region selection.

    Parameters
    ----------
    data_root : str
        Root directory containing Xenium data files.
    tissue_id : str
        Identifier for the tissue dataset.
    normalize_method : Optional[str]
        Method to normalize expression data. One of 'size', 'counts', or None.
    n_cells : Optional[int]
        Number of cells to select from a central spatial region.

    Returns
    -------
    AnnData
        Annotated data matrix with gene expression, cell metadata, and spatial coordinates.
    """
    if not os.path.exists(data_root):
        download_xenium_pancreas_cancer_data()

    feature_matrix = os.path.join(data_root, 'cell_feature_matrix.h5')
    cells_file = os.path.join(data_root, 'cells.csv.gz')
    types_file = os.path.join(data_root, 'Xenium_V1_hPancreas_Cancer_Add_on_FFPE_cell_groups.csv')

    # Read expression and metadata
    expr_adata = sc.read_10x_h5(feature_matrix)
    expr_df = pd.DataFrame(
        expr_adata.X.toarray(),
        index=expr_adata.obs_names.astype(str),
        columns=expr_adata.var_names
    )

    cells_df = pd.read_csv(cells_file, compression='gzip')
    types_df = pd.read_csv(types_file)

    # Ensure cell_id is string
    cells_df["cell_id"] = cells_df["cell_id"].astype(str)
    types_df["cell_id"] = types_df["cell_id"].astype(str)

    # Merge cell type info into cells
    merged = cells_df.merge(types_df, how="left", on="cell_id").set_index("cell_id")
    merged = merged.loc[expr_df.index]  # Align with expression data
    merged["cell_type"] = merged.get("group", "Unknown")

    # Spatial filtering by adaptive circular region
    if n_cells is not None:
        coords = merged[["x_centroid", "y_centroid"]].copy()

        # Step 1: Compute geometric center of the tissue
        center_x = coords["x_centroid"].mean()
        center_y = coords["y_centroid"].mean()

        # Step 2: Compute distance to center for each cell
        coords["dist_to_center"] = np.sqrt(
            (coords["x_centroid"] - center_x) ** 2 +
            (coords["y_centroid"] - center_y) ** 2
        )

        # Step 3: Select n closest cells
        selected = coords.sort_values("dist_to_center").iloc[:n_cells]
        selected_ids = selected.index

        # Step 4: Use farthest distance among selected cells as effective radius
        final_radius = selected["dist_to_center"].max()

        expr_df = expr_df.loc[selected_ids]
        merged = merged.loc[selected_ids]

        print(
            f"[INFO] Selected {n_cells} cells within radius "
            f"{final_radius:.2f} from center ({center_x:.2f}, {center_y:.2f})"
        )

    # Construct .obs
    obs = merged[["cell_type"]].copy()
    obs.index.name = "cell_id"
    obs["cell_area"] = merged.get("cell_area")
    obs["total_counts"] = expr_df.sum(axis=1)

    # Normalize if requested
    X = expr_df.loc[obs.index].values.astype(float)

    if normalize_method == "size" and obs["cell_area"].notna().all():
        X = X / obs["cell_area"].values[:, None]
    elif normalize_method == "counts":
        X = X / X.sum(axis=1, keepdims=True)

    # Build AnnData object
    adata = AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=expr_df.columns),
        obsm={"spatial": merged.loc[obs.index, ["x_centroid", "y_centroid"]].to_numpy()}
    )
    adata.uns["tissue_id"] = tissue_id
    adata.uns["data_level"] = "tissue"

    print(f"Successfully loaded {tissue_id} data: {adata}")
    return adata

def load_xenium_colorectal_cancer(
    data_root: str = os.path.join(DEFAULT_DATACACHE_DIR, "xenium_colorectal_cancer"),
    tissue_id: str = "Xenium_Human_Colorectal_Cancer",
) -> AnnData:
    """
    Load Xenium colorectal cancer data and prepare gene names and expression matrix.

    This function:
      - Ensures .X is correctly set from .layers["counts"]
      - Replaces invalid gene names in .var["gene_names"]
      - Sets .var_names to gene names (with fallback)
      - Adds default cell_type as 'spot'
    """
    if not os.path.exists(data_root):
        # download data
        download_xenium_colorectal_cancer_data(data_root)

    adata = sc.read_h5ad(f"{data_root}/adata.h5ad")

    # Sanitize gene names
    adata.var["gene_names"] = adata.var["gene_names"].astype(str)
    invalid = adata.var["gene_names"].isin(["nan", "NaN", "None", ""])
    adata.var.loc[invalid, "gene_names"] = adata.var_names[invalid]
    adata.var_names = adata.var["gene_names"].values
    adata.var_names_make_unique()

    # Set expression matrix
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"]
    else:
        raise ValueError("No expression matrix found in .X or .layers['counts'].")

    # Add metadata
    adata.obs["cell_type"] = "spot"
    adata.uns["tissue_id"] = tissue_id

    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.toarray()
    print(f"[✓] Successfully loaded {tissue_id} data: {adata}")
    return adata