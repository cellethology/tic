import pandas as pd

def get_cell_ids_by_type(region_id, cell_types, dataset_root):
    """
    Given a region_id and a list of cell types, retrieve the corresponding cell_ids.
    
    Args:
        region_id (str): The region ID.
        cell_types (list): List of cell types to filter by.
        dataset_root (str): The root directory of the dataset containing the region-level files.
    
    Returns:
        list: List of cell_ids corresponding to the given cell types.
    """
    # Path to the cell types CSV file for the specified region
    cell_types_file = f"{dataset_root}/voronoi/{region_id}.cell_types.csv"
    
    # Load the cell types data
    df = pd.read_csv(cell_types_file)
    
    # Ensure that the 'CELL_ID' and 'CELL_TYPE' columns are present
    if 'CELL_ID' not in df.columns or 'CLUSTER_LABEL' not in df.columns:
        raise ValueError(f"Columns 'CELL_ID' and 'CLUSTER_LABEL' are required in {cell_types_file}")
    
    # Filter cell ids based on the desired cell types
    filtered_df = df[df['CLUSTER_LABEL'].isin(cell_types)]
    
    # Return the list of cell_ids that match the specified cell types
    return filtered_df['CELL_ID'].tolist()
