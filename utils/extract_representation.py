# extract_representation.py
import os
import torch
from torch.utils.data import DataLoader
from core.data.dataset import MicroEDataset, MicroEWrapperDataset, collate_microe  

def get_region_ids_from_raw(root: str) -> list:
    raw_dir = os.path.join(root, "Raw")
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory not found at {raw_dir}")
    
    region_ids = list({fname.split('.')[0] for fname in os.listdir(raw_dir) if fname.endswith(".cell_data.csv")})
    
    return region_ids
# Main script for pseudo time analysis
def main():
    # Set the root directory and region IDs (modify paths as needed)
    root = "/Users/zhangjiahao/Dataset/CODEX/upmc/dataset"  # Adjust to your data root containing 'Raw' and 'Cache'
    region_ids = get_region_ids_from_raw(root)
    print(f"num of Tissue:{len(region_ids)}")
    
    # Create the MicroEDataset.
    dataset = MicroEDataset(
        root=root,
        region_ids=region_ids,
        k=3,
        microe_neighbor_cutoff=200.0,
        subset_cells=False,
        pre_transform = None,
        transform = None
    )
    
    # Wrap the dataset so that we get a list of MicroE objects.
    wrapper_ds = MicroEWrapperDataset(dataset)
    
    # Create a DataLoader that returns a list of MicroE objects per batch.
    dataloader = DataLoader(wrapper_ds, batch_size=1, shuffle=False, collate_fn=collate_microe)
    
    # List to hold the center cells with exported representations.
    center_cells = []
    
    # Iterate over the DataLoader.
    for batch in dataloader:
        # Since batch_size=1, batch is a list with one MicroE object.
        for microe in batch:
            # Export the center cell with representations.
            # This method will add features (like raw_expression, neighbor_composition, etc.)
            # to the center cell and return the updated Cell.
            center_cell = microe.export_center_cell_with_representations()
            center_cells.append(center_cell)
    
    print(f"Extracted {len(center_cells)} center cells for pseudotime analysis.")
    
    # Optionally, save the list of center cells to disk for later analysis.
    save_path = os.path.join(root, "center_cells.pt")
    torch.save(center_cells, save_path)
    print(f"Saved center cell representations to {save_path}")

if __name__ == "__main__":
    main()