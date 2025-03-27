import torch
import random

from tic.constant import ALL_BIOMARKERS

def mask_biomarker_expression(data, mask_ratio=0.2):
    """
    Mask a portion of the biomarker_expression for a given data object.
    
    Args:
        data (Data): PyG Data object containing 'x' (node features) and 'y' (ground truth).
        mask_ratio (float): Ratio of biomarkers to mask (default is 0.2, i.e., 20% masked).
    
    Returns:
        Data: Updated PyG Data object with masked biomarker expression.
    """
    # Get the biomarker feature indices
    biomarker_start, biomarker_end = data.feature_indices['biomarker_expression']
    
    # Store the original biomarker values in y (ground truth)
    data.y = data.x[:, biomarker_start:biomarker_end]  # The original values as ground truth

    # Masking the biomarker expressions by setting them to 0 (or any other value you wish)
    num_biomarkers = biomarker_end - biomarker_start  # Calculate the number of biomarkers to mask
    num_masked = int(mask_ratio * num_biomarkers)  # Number of biomarkers to mask

    # Create a mask tensor of size (num_biomarkers) initialized to 0
    mask_tensor = torch.zeros(num_biomarkers, dtype=torch.long)

    # Randomly select indices of biomarkers to mask
    masked_indices = random.sample(range(num_biomarkers), num_masked)
    
    # Set the selected indices to 1 in the mask tensor (indicating these biomarkers are masked)
    mask_tensor[masked_indices] = 1
    
    # Apply the mask by setting the corresponding biomarker values to 0
    biomarker_expr = data.x[:, biomarker_start:biomarker_end]
    biomarker_expr[:, mask_tensor.bool()] = 0  # Masked values set to 0
    
    # Store the mask tensor in the data object
    data.mask = mask_tensor
    
    return data



# transform function 

def mask_transform(data):
    """
    Transform function for masked learning tasks.
    
    1. Sets the target `y` by copying the last len(ALL_BIOMARKERS) dimensions
       of the center cell's (assumed to be the first node) feature vector.
    2. Applies random masking to the last len(ALL_BIOMARKERS) dimensions of 
       all nodes' features with a masking ratio uniformly sampled from 0 to 0.2.
       
    :param data: A PyG Data object with attribute `x` (node features).
    :return: The modified Data object with new attributes `x` (masked) and `y` (target).
    """
    num_biomarkers = len(ALL_BIOMARKERS)
    
    # Ensure node features are a tensor (if not already)
    if not isinstance(data.x, torch.Tensor):
        data.x = torch.tensor(data.x, dtype=torch.float)
    
    # Copy the last num_biomarkers dimensions of the center cell's features as target y.
    # In MicroE, center cell is always at index 0
    data.y = data.x[0, -num_biomarkers:].clone()
    
    # Randomly determine a masking ratio between 0 and 0.2
    mask_ratio = random.uniform(0, 0.2)
    
    # For all nodes, apply the masking to the last num_biomarkers features.
    num_nodes = data.x.size(0)
    biomarker_features = data.x[:, -num_biomarkers:]
    
    # Create a mask tensor where each element is 1 with probability (1 - mask_ratio)
    # and 0 with probability mask_ratio.
    mask = (torch.rand(num_nodes, num_biomarkers) > mask_ratio).float()
    
    # Apply the mask to the biomarker features
    data.x[:, -num_biomarkers:] = biomarker_features * mask

    data.mask = mask[0,:].bool() 
    data.mask = ~data.mask # true for masked biomarkers, false for known biomarkers
    return data
