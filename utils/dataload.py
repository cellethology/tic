# utils/dataload.py

import os
import anndata
import pandas as pd
from tic.constant import FILE_MAPPING

def process_region_to_anndata(
    raw_dir: str, 
    region_id: str,
    csv_file_mapping: dict = FILE_MAPPING,
    required_columns: dict = None,
) -> anndata.AnnData:
    """
    Reads multiple CSV files corresponding to a single region and merges the data
    into an AnnData object.
    
    :param raw_dir: Directory containing raw CSV files.
    :param region_id: Identifier for the region (used for filename matching).
    :param csv_file_mapping: A mapping of file types to filename templates.
           For example:
           {
             'coords': "{region_id}.cell_data.csv",
             'features': "{region_id}.cell_features.csv",
             'types': "{region_id}.cell_types.csv",
             'expression': "{region_id}.expression.csv"
           }
    :param required_columns: Dictionary of required columns for each file type.
           If None, defaults are used:
           {
             'coords': ['CELL_ID', 'X', 'Y'],
             'features': ['CELL_ID', 'SIZE'],
             'types': ['CELL_ID', 'CELL_TYPE'],
             'expression': ['CELL_ID']
           }
    :return: An AnnData object with:
             - X: biomarker expression matrix,
             - obs: cell metadata (cell type, size, etc.),
             - obsm["spatial"]: spatial coordinates,
             - var: biomarker annotations.
    """
    if required_columns is None:
        required_columns = {
            'coords': ['CELL_ID', 'X', 'Y'],
            'features': ['CELL_ID', 'SIZE'],
            'types': ['CELL_ID', 'CELL_TYPE'],
            'expression': ['CELL_ID']  # The remaining columns are biomarkers.
        }
    
    # Build paths for each CSV file.
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
    
    # Merge the DataFrames on 'CELL_ID'.
    df = (
        coords_df.merge(features_df, on="CELL_ID")
                 .merge(types_df, on="CELL_ID")
                 .merge(expression_df, on="CELL_ID")
    )
    
    # Extract biomarker expression matrix from expression columns.
    biomarker_columns = [col for col in expression_df.columns if col != 'CELL_ID']
    X = df[biomarker_columns].to_numpy()
    
    # Create obs DataFrame with metadata.
    obs = df[['CELL_ID', 'CELL_TYPE', 'SIZE']].copy().set_index('CELL_ID')
    
    # Build obsm with spatial coordinates.
    spatial_coords = df[['X', 'Y']].to_numpy()
    obsm = {"spatial": spatial_coords}
    
    # Create var DataFrame (here simply using biomarker names).
    var = pd.DataFrame(index=biomarker_columns)
    
    # Create and return AnnData.
    adata = anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)
    adata.uns["region_id"] = region_id
    return adata