"""
Module: tic.model.transform
Provides functions to perform masking transformations on biomarker expression data.
"""

import torch
import random
from tic.constant import ALL_BIOMARKERS


def mask_biomarker_expression(data, mask_ratio: float = 0.2):
    """
    Mask a portion of the biomarker expression in a PyG Data object.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Data object containing 'x' (node features) and a dictionary attribute 'feature_indices'
        with key 'biomarker_expression'.
    mask_ratio : float, optional
        Ratio of biomarkers to mask (default 0.2).

    Returns
    -------
    torch_geometric.data.Data
        Updated Data object with biomarker expressions masked and original values stored in 'y'.
    """
    biomarker_start, biomarker_end = data.feature_indices["biomarker_expression"]
    data.y = data.x[:, biomarker_start:biomarker_end].clone()

    num_biomarkers = biomarker_end - biomarker_start
    num_masked = int(mask_ratio * num_biomarkers)

    mask_tensor = torch.zeros(num_biomarkers, dtype=torch.long)
    masked_indices = random.sample(range(num_biomarkers), num_masked)
    mask_tensor[masked_indices] = 1

    biomarker_expr = data.x[:, biomarker_start:biomarker_end]
    biomarker_expr[:, mask_tensor.bool()] = 0

    data.mask = mask_tensor
    return data


def mask_transform(data):
    """
    Apply masking transformation for masked learning tasks.

    1. Sets the target 'y' as the center cell's biomarker expression (last len(ALL_BIOMARKERS) features).
    2. Applies random masking to the biomarker features of all nodes.
    3. Stores the mask indicating masked biomarkers.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Data object with attribute 'x' containing node features.

    Returns
    -------
    torch_geometric.data.Data
        Modified Data object with new attributes 'x' (masked), 'y' (target), and 'mask'.
    """
    num_biomarkers = len(ALL_BIOMARKERS)
    if not isinstance(data.x, torch.Tensor):
        data.x = torch.tensor(data.x, dtype=torch.float)

    # Set target from the center cell (first node)
    data.y = data.x[0, -num_biomarkers:].clone()

    mask_ratio = random.uniform(0, 0.2)
    num_nodes = data.x.size(0)
    biomarker_features = data.x[:, -num_biomarkers:]
    mask = (torch.rand(num_nodes, num_biomarkers) > mask_ratio).float()

    data.x[:, -num_biomarkers:] = biomarker_features * mask
    # For the center cell: True for masked biomarkers, False for observed ones.
    center_mask = mask[0, :].bool()
    data.mask = ~center_mask
    return data