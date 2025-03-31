"""
Module: tic.model.feature
Provides functions to process and pre-transform biomarker expression data.
"""

import numpy as np
from scipy.stats import rankdata
import torch

from tic.constant import ALL_BIOMARKERS


def process_biomarker_expression(
    biomarker_expr_list, method: str = "rank", lb: float = 0, ub: float = 1
) -> np.ndarray:
    """
    Process biomarker expression data based on the selected method.

    Parameters
    ----------
    biomarker_expr_list : list or np.ndarray
        List or array of biomarker expression values.
    method : str, optional
        Processing method ('rank', 'linear', 'log', or 'raw'). Default is 'rank'.
    lb : float, optional
        Lower bound for normalization. Default is 0.
    ub : float, optional
        Upper bound for normalization. Default is 1.

    Returns
    -------
    np.ndarray
        Processed biomarker expression values.
    """
    biomarker_expr_array = np.array(biomarker_expr_list)

    if method == "rank":
        ranked_expr = rankdata(biomarker_expr_array, method="min")
        num_exp = len(ranked_expr)
        ranked_expr = (ranked_expr - 1) / (num_exp - 1)
    elif method == "linear":
        ranked_expr = np.clip(biomarker_expr_array, lb, ub)
        ranked_expr = (ranked_expr - lb) / (ub - lb)
    elif method == "log":
        ranked_expr = np.clip(np.log(biomarker_expr_array + 1e-9), lb, ub)
        ranked_expr = (ranked_expr - lb) / (ub - lb)
    elif method == "raw":
        ranked_expr = biomarker_expr_array
    else:
        raise ValueError(f"Expression process method {method} not recognized")
    return ranked_expr


def biomarker_pretransform(data, method: str = "rank", lb: float = 0, ub: float = 1):
    """
    Pre-transform biomarker expression data in a PyG Data object.

    This function:
      1. Extracts the last len(ALL_BIOMARKERS) dimensions from each node's feature vector.
      2. Processes these values using `process_biomarker_expression`.
      3. Replaces the original biomarker expression values in data.x with the processed values.

    Parameters
    ----------
    data : torch_geometric.data.Data
        PyG Data object containing node features in attribute `x`.
    method : str, optional
        Processing method to use (default is 'rank').
    lb : float, optional
        Lower bound for normalization (default is 0).
    ub : float, optional
        Upper bound for normalization (default is 1).

    Returns
    -------
    torch_geometric.data.Data
        Updated Data object with processed biomarker expression values.
    """
    num_biomarkers = len(ALL_BIOMARKERS)

    if not isinstance(data.x, torch.Tensor):
        data.x = torch.tensor(data.x, dtype=torch.float)

    biomarker_values = data.x[:, -num_biomarkers:].cpu().numpy()

    processed_values = np.array(
        [
            process_biomarker_expression(row, method=method, lb=lb, ub=ub)
            for row in biomarker_values
        ]
    )

    new_x = data.x.clone()
    new_x[:, -num_biomarkers:] = torch.tensor(
        processed_values, dtype=new_x.dtype, device=new_x.device
    )
    data.x = new_x
    return data