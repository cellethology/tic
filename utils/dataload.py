# utils/dataload.py

import os
import anndata
import pandas as pd
from tic.constant import FILE_MAPPING
from tic.data.graph_feature import edge_index_fn, node_feature_fn
from tic.data.tissue import Tissue
from utils.microe2ann import export_center_cells

def process_region_to_anndata(
    raw_dir: str, 
    region_id: str,
    csv_file_mapping: dict = FILE_MAPPING,
    required_columns: dict = None,
) -> anndata.AnnData:
    """
    Reads multiple CSV files corresponding to a single region and merges the data
    into a standardized AnnData object.

    The output AnnData follows these conventions:
      - X: A biomarker expression matrix where each row corresponds to a cell and 
           each column corresponds to a biomarker.
      - obs: A DataFrame of cell metadata with the index set to "CELL_ID". The columns include:
             - cell_type: The cell's type (converted from 'CELL_TYPE').
             - size: The cell's size (converted from 'SIZE').
             - (Any additional features from the merged CSVs.)
      - var: A DataFrame whose index consists of biomarker names (all columns in expression CSV except 'CELL_ID').
      - obsm: A dictionary with key "spatial" storing the spatial coordinates (from 'X' and 'Y' columns).
      - uns: A dictionary that stores additional metadata, including "region_id" and "data_level" set to "tissue".

    :param raw_dir: Directory containing raw CSV files.
    :param region_id: Identifier for the region (used for filename matching).
    :param csv_file_mapping: A mapping of file types to filename templates. For example:
           {
             'coords': "{region_id}.cell_data.csv",
             'features': "{region_id}.cell_features.csv",
             'types': "{region_id}.cell_types.csv",
             'expression': "{region_id}.expression.csv"
           }
    :param required_columns: Dictionary specifying required columns for each file type.
           If None, defaults are used:
           {
             'coords': ['CELL_ID', 'X', 'Y'],
             'features': ['CELL_ID', 'SIZE'],
             'types': ['CELL_ID', 'CELL_TYPE'],
             'expression': ['CELL_ID']
           }
    :return: An AnnData object with:
             - X: biomarker expression matrix,
             - obs: cell metadata (cell_id, cell_type, size, etc.),
             - obsm["spatial"]: spatial coordinates,
             - var: biomarker annotations,
             - uns: data_level and tissue_id
    """
    if required_columns is None:
        required_columns = {
            'coords': ['CELL_ID', 'X', 'Y'],
            'features': ['CELL_ID', 'SIZE'],
            'types': ['CELL_ID', 'CELL_TYPE'],
            'expression': ['CELL_ID']  # The remaining columns are biomarkers.
        }
    
    # Construct file paths.
    coords_path = os.path.join(raw_dir, csv_file_mapping['coords'].format(region_id=region_id))
    features_path = os.path.join(raw_dir, csv_file_mapping['features'].format(region_id=region_id))
    types_path = os.path.join(raw_dir, csv_file_mapping['types'].format(region_id=region_id))
    expression_path = os.path.join(raw_dir, csv_file_mapping['expression'].format(region_id=region_id))
    
    # Read CSV files.
    coords_df = pd.read_csv(coords_path)
    features_df = pd.read_csv(features_path)
    types_df = pd.read_csv(types_path)
    expression_df = pd.read_csv(expression_path)
    
    # Optionally drop unwanted columns.
    drop_cols = ['ACQUISITION_ID']
    for col in drop_cols:
        if col in expression_df.columns:
            expression_df = expression_df.drop(columns=[col])
    
    # Merge DataFrames on 'CELL_ID'.
    df = (
        coords_df.merge(features_df, on="CELL_ID")
                 .merge(types_df, on="CELL_ID")
                 .merge(expression_df, on="CELL_ID")
    )
    
    # Extract biomarker expression matrix from expression columns (excluding 'CELL_ID').
    biomarker_columns = [col for col in expression_df.columns if col != 'CELL_ID']
    X = df[biomarker_columns].to_numpy()
    
    # Create obs DataFrame with metadata; standardize column names.
    obs = df[['CELL_ID', 'CELL_TYPE', 'SIZE']].copy()
    obs.rename(columns={'CELL_TYPE': 'cell_type', 'SIZE': 'size', 'CELL_ID': 'cell_id'}, inplace=True)
    
    # Build obsm with spatial coordinates (using 'X' and 'Y' columns).
    spatial_coords = df[['X', 'Y']].to_numpy()
    obsm = {"spatial": spatial_coords}
    
    # Build var DataFrame using biomarker names.
    var = pd.DataFrame(index=biomarker_columns)
    
    # Create and return the AnnData object.
    adata = anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)
    adata.uns["tissue_id"] = region_id
    adata.uns["data_level"] = "tissue"
    return adata

if __name__ == "__main__":
    example_dir = "data/example/Raw"
    example_region = "UPMC_c001_v001_r001_reg001"
    tissue_ann = process_region_to_anndata(raw_dir=example_dir,region_id=example_region)
    print(tissue_ann)
    tissue = Tissue.from_anndata(tissue_ann)
    tissue_ann_exported = tissue.to_anndata()
    tissue.to_graph(edge_index_fn=edge_index_fn, node_feature_fn=node_feature_fn)
    microe = tissue.get_microenvironment(center_cell_id='1')
    microe_ann = microe.to_anndata()
    c_cell = microe.center_cell
    cell_ann = c_cell.to_anndata()
    print(cell_ann)
    print(microe_ann)
    print(tissue_ann_exported)

    center_ann = export_center_cells([microe],representations= ["raw_expression", "neighbor_composition"])
    print(center_ann)
    print(center_ann.var)