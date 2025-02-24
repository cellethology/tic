import torch
import random

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
