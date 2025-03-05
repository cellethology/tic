# utils/extract_representation.py
import os
from typing import List
import torch
from torch.utils.data import DataLoader
from core.data.dataset import MicroEDataset, MicroEWrapperDataset, collate_microe

def get_region_ids_from_raw(root: str) -> List[str]:
    """
    Retrieve region IDs from the "Raw" directory under the given root.

    The function scans the directory for files ending with ".cell_data.csv"
    and extracts the region IDs based on the filename (using the part before the dot).

    Args:
        root (str): The root directory that should contain a "Raw" folder.

    Returns:
        List[str]: A list of region IDs.
    
    Raises:
        FileNotFoundError: If the "Raw" directory does not exist.
    """
    raw_dir = os.path.join(root, "Raw")
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory not found at {raw_dir}")
    region_ids = list({fname.split('.')[0] for fname in os.listdir(raw_dir) if fname.endswith(".cell_data.csv")})
    return region_ids

def extract_center_cells(root: str,
                         k: int = 3,
                         microe_neighbor_cutoff: float = 200.0,
                         batch_size: int = 1) -> list:
    """
    Extract center cell representations from the dataset.

    This function loads the dataset using provided parameters, wraps it, and iterates
    through the data loader to export center cell representations with additional features.

    Args:
        root (str): The root directory of the dataset, which must include "Raw" and "Cache" folders.
        k (int, optional): The neighbor parameter. Defaults to 3.
        microe_neighbor_cutoff (float, optional): Cutoff distance for neighborhood. Defaults to 200.0.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 1.

    Returns:
        list: A list of center cell objects with representations.
    """
    region_ids = get_region_ids_from_raw(root)
    print(f"[Info] Found {len(region_ids)} regions.")

    dataset = MicroEDataset(
        root=root,
        region_ids=region_ids,
        k=k,
        microe_neighbor_cutoff=microe_neighbor_cutoff,
        subset_cells=False,
        pre_transform=None,
        transform=None
    )
    wrapper_ds = MicroEWrapperDataset(dataset)
    dataloader = DataLoader(wrapper_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_microe)

    center_cells = []
    for batch in dataloader:
        for microe in batch:
            center_cell = microe.export_center_cell_with_representations()
            center_cells.append(center_cell)

    print(f"[Info] Extracted {len(center_cells)} center cells for pseudotime analysis.")
    return center_cells

def save_center_cells(center_cells: list, save_path: str) -> None:
    """
    Save the list of center cells to disk.

    Args:
        center_cells (list): List of center cell objects.
        save_path (str): File path to save the center cells.
    """
    torch.save(center_cells, save_path)
    print(f"[Info] Saved center cell representations to {save_path}")

def main():
    """
    Main function to extract and save center cell representations.
    """
    # Example parameters (adjust the root path as needed)
    root = "../../data/example"
    save_path = os.path.join(root, "center_cells.pt")
    
    center_cells = extract_center_cells(root=root)
    save_center_cells(center_cells, save_path=save_path)

if __name__ == "__main__":
    main()