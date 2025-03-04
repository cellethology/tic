import numpy as np
from scipy.stats import rankdata
import torch

from core.constant import ALL_BIOMARKERS

def process_biomarker_expression(biomarker_expr_list, method='rank', lb=0, ub=1):
    """
    Processes the biomarker expression data based on the selected method.

    Args:
        biomarker_expr_list (list or np.array): List or array of biomarker expression values.
        method (str): Processing method. Default is 'rank'.
        lb (float): Lower bound for normalization.
        ub (float): Upper bound for normalization.

    Returns:
        np.array: Processed biomarker expression values.
    """
    # Convert to numpy array if the input is a list
    biomarker_expr_array = np.array(biomarker_expr_list)
    
    if method == 'rank':
        # Apply ranking to biomarker expression
        ranked_expr = rankdata(biomarker_expr_array, method='min')
        num_exp = len(ranked_expr)
        ranked_expr = (ranked_expr - 1) / (num_exp - 1)  # Normalize between 0 and 1

    elif method == 'linear':
        # Apply linear transformation between lb and ub
        ranked_expr = np.clip(biomarker_expr_array, lb, ub)
        ranked_expr = (ranked_expr - lb) / (ub - lb)

    elif method == 'log':
        # Apply log transformation and normalization
        ranked_expr = np.clip(np.log(biomarker_expr_array + 1e-9), lb, ub)
        ranked_expr = (ranked_expr - lb) / (ub - lb)

    elif method == 'raw':
        # No transformation, return original data
        ranked_expr = biomarker_expr_array

    else:
        raise ValueError(f"Expression process method {method} not recognized")
    
    return ranked_expr

def biomarker_pretransform(data, method='rank', lb=0, ub=1):
    """
    Pre-transform function to process biomarker expression data in the PyG Data object.

    This function:
      1. Extracts the last len(ALL_BIOMARKERS) dimensions from each node's feature vector,
         which correspond to the biomarker expression values.
      2. Processes these values using the `process_biomarker_expression` function.
      3. Replaces the original biomarker expression values in data.x with the processed values.

    Args:
        data (torch_geometric.data.Data): The PyG Data object containing node features in attribute `x`.
        method (str): The processing method to use (default is 'rank').
        lb (float): Lower bound for normalization (default is 0).
        ub (float): Upper bound for normalization (default is 1).

    Returns:
        torch_geometric.data.Data: The updated Data object with processed biomarker expression values.
    """
    num_biomarkers = len(ALL_BIOMARKERS)
    
    # Ensure that data.x is a torch.Tensor
    if not isinstance(data.x, torch.Tensor):
        data.x = torch.tensor(data.x, dtype=torch.float)
    
    # Extract the biomarker expression values (assumed to be in the last num_biomarkers columns)
    biomarker_values = data.x[:, -num_biomarkers:].cpu().numpy()
    
    # Process each node's biomarker expression independently.
    processed_values = []
    for row in biomarker_values:
        processed_row = process_biomarker_expression(row, method=method, lb=lb, ub=ub)
        processed_values.append(processed_row)
    
    processed_values = np.array(processed_values)
    
    # Replace the original biomarker expression part with the processed values.
    new_x = data.x.clone()
    new_x[:, -num_biomarkers:] = torch.tensor(processed_values, dtype=new_x.dtype, device=new_x.device)
    data.x = new_x
    
    return data