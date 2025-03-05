import os
import torch
from torch.utils.data import DataLoader
from core.data.dataset import MicroEDataset, MicroEWrapperDataset, collate_microe  


# Main script for pseudo time analysis
def main():
    # Set the root directory and region IDs (modify paths as needed)
    root = "/Users/zhangjiahao/Project/tic/data/example/"  # Adjust to your data root containing 'Raw' and 'Cache'
    region_ids = ["UPMC_c001_v001_r001_reg001", "UPMC_c001_v001_r001_reg004"]  # List your region/tissue IDs
    
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