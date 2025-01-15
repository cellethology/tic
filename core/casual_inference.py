# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:03 2025

@author: Jiahao Zhang
@Description: 
"""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from adapters.space_gm_adapter import get_neighborhood_cell_ids
from statsmodels.tsa.stattools import grangercausalitytests
from tqdm import tqdm
#----------------------------------
# Helper Functions 
#----------------------------------

def load_pseudotime(pseudotime_file):
    """
    Load pseudo-time data from a CSV file.

    Args:
        pseudotime_file (str): Path to the pseudo-time CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns ['region_id', 'cell_id', 'pseudotime'].
    """
    pseudotime_df = pd.read_csv(pseudotime_file)
    return pseudotime_df

def compute_biomarker_matrix(dataset, region_id, center_cell_id, raw_dir, cell_type_mapping):
    """
    Compute the N x M matrix of biomarker averages for neighborhood cells.

    Args:
        dataset (CellularGraphDataset): The dataset instance.
        region_id (str): The region ID to search for.
        center_cell_id (int): The center cell ID.
        raw_dir (str): Path to the raw data directory.
        cell_type_mapping (dict): Mapping of all possible cell types (keys as strings).

    Returns:
        pd.DataFrame: A DataFrame representing the N x M biomarker matrix, indexed by cell type names (str).
    """
    # Step 1: Retrieve neighborhood cell IDs
    neighborhood_cell_ids = get_neighborhood_cell_ids(dataset, region_id, center_cell_id)
    
    # Step 2: Load necessary data
    cell_types_file = f"{raw_dir}/{region_id}.cell_types.csv"
    expression_file = f"{raw_dir}/{region_id}.expression.csv"
    
    cell_types_df = pd.read_csv(cell_types_file)
    expression_df = pd.read_csv(expression_file)
    
    # Step 3: Filter for neighborhood cell IDs
    neighborhood_df = cell_types_df[cell_types_df["CELL_ID"].isin(neighborhood_cell_ids)].copy()
    expression_subset = expression_df[expression_df["CELL_ID"].isin(neighborhood_cell_ids)].copy()
    
    # Step 4: Merge cell types with biomarker expressions
    merged_df = pd.merge(neighborhood_df, expression_subset, on="CELL_ID")
    
    # Step 5: Rename `CLUSTER_LABEL` to `CELL_TYPE` if necessary
    if "CLUSTER_LABEL" in merged_df.columns:
        merged_df.rename(columns={"CLUSTER_LABEL": "CELL_TYPE"}, inplace=True)
    
    # Step 6: Compute the average biomarker values per cell type
    biomarker_columns = [col for col in merged_df.columns if col not in ["CELL_ID", "REGION_ID", "CELL_TYPE", "ACQUISITION_ID"]]
    biomarker_matrix = merged_df.groupby("CELL_TYPE")[biomarker_columns].mean()
    
    # Step 7: Reindex to include all cell types from `cell_type_mapping` keys
    all_cell_types = list(cell_type_mapping.keys())  # Use the keys (str) from the mapping
    biomarker_matrix = biomarker_matrix.reindex(all_cell_types, fill_value=0)
    
    return biomarker_matrix

def prepare_granger_inputs(dataset, raw_dir, pseudotime_file, cell_type_mapping, max_workers=4):
    """
    Prepare data for Granger causality analysis.

    Args:
        dataset (CellularGraphDataset): The dataset instance.
        raw_dir (str): Path to the raw data directory.
        pseudotime_file (str): Path to the pseudo-time CSV file.
        cell_type_mapping (dict): Mapping of all possible cell types (keys as strings).
        max_workers (int): Number of threads for multithreading.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: 1D array of pseudo-time values (length S).
            - np.ndarray: 3D array of biomarker matrices (shape S x N x M).
    """
    # Step 1: Load pseudo-time data
    pseudotime_df = pd.read_csv(pseudotime_file)

    # Validate columns
    if not {"region_id", "cell_id", "pseudotime"}.issubset(pseudotime_df.columns):
        raise ValueError("The pseudo-time file must contain 'region_id', 'cell_id', and 'pseudotime' columns.")

    # Step 2: Initialize storage for biomarker matrices and pseudo-time values
    pseudo_time_values = pseudotime_df["pseudotime"].to_numpy()
    biomarker_matrices = []

    # Step 3: Define a worker function for parallel computation
    def process_cell(row):
        region_id = row["region_id"]
        cell_id = row["cell_id"]
        return compute_biomarker_matrix(
            dataset=dataset,
            region_id=region_id,
            center_cell_id=cell_id,
            raw_dir=raw_dir,
            cell_type_mapping=cell_type_mapping
        )

    # Step 4: Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_cell, row)
            for _, row in pseudotime_df.iterrows()
        ]

        # Collect results
        for future in tqdm(futures, total=len(futures), desc="Processing cells"):
            try:
                biomarker_matrix = future.result()
                biomarker_matrices.append(biomarker_matrix.to_numpy())
            except Exception as e:
                print(f"Error processing a cell: {e}")
                biomarker_matrices.append(np.zeros((len(cell_type_mapping), len(biomarker_matrix.columns))))

    # Step 5: Convert biomarker matrices to a single 3D array
    neighborhood_matrices = np.array(biomarker_matrices)  # Shape: (S x N x M)
    cell_type_names = list(cell_type_mapping.keys())  
    biomarker_names = [col for col in biomarker_matrix.columns]  

    return pseudo_time_values, neighborhood_matrices, cell_type_names, biomarker_names

#----------------------------------
# Core: Casual Inference
# Key Input: pseudo_time (np.ndarray): 1D array of pseudo-time (length S).
#            neighborhood_matrices (np.ndarray): 3D array of neighborhood expression matrices (shape S x N x M).
#----------------------------------

def compute_granger_causality(pseudo_time, neighborhood_matrices, max_lag=3, significance_level=0.05, cell_type_names = None, biomarker_names = None):
    """
    Perform Granger causality analysis between pseudo-time and neighborhood expression components.
    To Visualize the results, you can use the following function:
        from utils.visualization import visualize_granger_results
    Args:
        pseudo_time (np.ndarray): 1D array of pseudo-time (length S).
        neighborhood_matrices (np.ndarray): 3D array of neighborhood expression matrices (shape S x N x M).
        max_lag (int): Maximum number of lags for Granger causality test.
        significance_level (float): Significance level for the test.

    Returns:
        pd.DataFrame: DataFrame with Granger causality results (N x M).
    """
    S, N, M = neighborhood_matrices.shape
    results_matrix = np.zeros((N, M))  # To store p-values

    # Flatten neighborhood matrices to (S, N*M)
    flattened_matrices = neighborhood_matrices.reshape(S, N * M)

    for idx in range(flattened_matrices.shape[1]):  # Iterate over N*M components
        component_series = flattened_matrices[:, idx]

        # Combine pseudo_time and component_series into a DataFrame
        data = np.column_stack((pseudo_time, component_series))
        data = pd.DataFrame(data, columns=["pseudo_time", f"component_{idx}"])

        try:
            # Perform Granger causality test
            test_result = grangercausalitytests(data, max_lag, verbose=False)
            min_p_value = min([test_result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)])
        except Exception as e:
            print(f"Error in Granger test for component {idx}: {e}")
            min_p_value = 1.0

        # Store p-value in the results matrix
        results_matrix[idx // M, idx % M] = min_p_value

    # Apply significance threshold
    significant_matrix = (results_matrix < significance_level).astype(int)

    return pd.DataFrame(
        results_matrix, 
        index=[f"CellType_{i}" for i in range(N)] if cell_type_names is None else cell_type_names,
        columns=[f"Biomarker_{j}" for j in range(M)] if biomarker_names is None else biomarker_names
    ), pd.DataFrame(
        significant_matrix, 
        index=[f"CellType_{i}" for i in range(N)] if cell_type_names is None else cell_type_names,
        columns=[f"Biomarker_{j}" for j in range(M)] if biomarker_names is None else biomarker_names
    )

