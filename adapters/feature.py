import numpy as np
from scipy.stats import rankdata

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