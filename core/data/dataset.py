import os
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset
from typing import List

from core.data.graph_feature import edge_attr_fn, edge_index_fn, node_feature_fn
from core.model.feature import biomarker_pretransform
from core.model.transform import mask_transform
from utils.dataload import process_region_to_tissue

class MicroEDataset(InMemoryDataset):
    """
    A PyG InMemoryDataset for generating microenvironment-level Data objects for GNN training.
    It builds Tissue objects from raw single-cell CSV data, extracts MicroE subgraphs around each cell,
    and returns a Data object with associated region_id and cell_id. The dataset supports filtering 
    the center cell based on specified cell types.

    Additionally, this version saves each raw (untransformed) MicroE graph as a .pt file
    before any pre_transform is applied.
    """

    def __init__(self,
                 root: str,
                 region_ids: List[str],
                 k: int = 3,
                 transform=None,
                 pre_transform=None,
                 microe_neighbor_cutoff: float = 200.0,
                 subset_cells: bool = False,
                 center_cell_types: List[str] = ["Tumor"]):
        """
        :param root: Root directory, should contain 'Raw/' and 'Cache/' subfolders.
        :param region_ids: List of region/tissue IDs.
        :param k: k-hop neighborhood for extracting MicroE subgraphs.
        :param transform: Optional PyG transform.
        :param pre_transform: Optional PyG pre_transform.
        :param microe_neighbor_cutoff: Distance threshold for filtering neighbors.
        :param subset_cells: If True, sample a subset of cells for large tissues.
        :param center_cell_types: List of cell types to consider as center cells for MicroE extraction.
        """
        self.root = root
        self.region_ids = region_ids
        self.k = k
        self.microe_neighbor_cutoff = microe_neighbor_cutoff
        self.subset_cells = subset_cells
        self.center_cell_types = center_cell_types
        super().__init__(root, transform, pre_transform)
        
        # After processing, load the collated dataset from disk:
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        raw_dir_path = os.path.join(self.root, "Raw")
        if not os.path.exists(raw_dir_path):
            raise FileNotFoundError(f"Raw data directory not found at {raw_dir_path}")
        return raw_dir_path

    @property
    def processed_dir(self):
        processed_dir_path = os.path.join(self.root, "Cache")
        if not os.path.exists(processed_dir_path):
            os.makedirs(processed_dir_path)
        return processed_dir_path

    @property
    def raw_file_names(self):
        # List of expected raw CSV file names for each region.
        files = []
        for rid in self.region_ids:
            files.extend([
                f"{rid}.cell_data.csv",
                f"{rid}.cell_features.csv",
                f"{rid}.cell_types.csv",
                f"{rid}.expression.csv"
            ])
        return files

    @property
    def processed_file_names(self):
        """
        Single processed file combining all microenvironment graphs after optional pre_transform.
        Raw micrographs are saved individually in .pt files, but this is the 'final' dataset file.
        """
        fname = f"microe_dataset_{len(self.region_ids)}_k{self.k}_cutoff{self.microe_neighbor_cutoff}.pt"
        return [fname]

    def download(self):
        # No auto-download implemented; assume data is local under raw_dir.
        pass

    def process(self):
        """
        Process raw CSV files to generate microenvironment subgraphs:
          1) Load or build the Tissue object
          2) Convert Tissue to a graph (if not done)
          3) Extract MicroE subgraphs for each center cell
          4) Save the raw micro_graph (before pre_transform) to .pt 
          5) If pre_transform is defined, transform the micro_graph
          6) Collect all final Data objects into an InMemoryDataset
        """
        data_list = []

        for rid in self.region_ids:
            print(f"[MicroEDataset] Processing Tissue {rid} ...")

            # 1) Load or build Tissue
            tissue_cache_path = os.path.join(self.processed_dir, f"Tissue_{rid}.pt")
            if os.path.exists(tissue_cache_path):
                tissue = torch.load(tissue_cache_path)
            else:
                tissue = process_region_to_tissue(self.raw_dir, rid)
                tissue.to_graph(node_feature_fn, edge_index_fn, edge_attr_fn)
                torch.save(tissue, tissue_cache_path)

            # Filter center cells by type
            cell_list = [c for c in tissue.cells if c.cell_type in self.center_cell_types]

            # Optionally subset for large tissues
            if self.subset_cells:
                cell_list = np.random.choice(cell_list, size=min(100, len(cell_list)), replace=False)

            for cell in cell_list:
                center_id = cell.cell_id
                micro_env = tissue.get_microenvironment(
                    center_id,
                    k=self.k,
                    microe_neighbor_cutoff=self.microe_neighbor_cutoff
                )
                micro_graph = micro_env.graph  # A PyG Data object

                # 2) Save the raw micro_graph before any pre_transform
                raw_micro_graph_path = os.path.join(self.processed_dir, f"MicroE_{rid}_{center_id}.pt")
                torch.save(micro_graph, raw_micro_graph_path)

                # 3) If pre_transform is provided, apply it now
                if self.pre_transform is not None:
                    micro_graph = self.pre_transform(micro_graph)

                # Attach region/cell meta info:
                micro_graph.region_id = rid
                micro_graph.cell_id = center_id

                data_list.append(micro_graph)

        # Collate all micro_graph data into a single file for the InMemoryDataset
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def len(self):
        # Number of microenvironment subgraphs in the final dataset
        return self.slices['x'].size(0) - 1

if __name__ == "__main__":
    dataset = MicroEDataset(
        root="/Users/zhangjiahao/Project/tic/data/example",
        region_ids=["UPMC_c001_v001_r001_reg001", "UPMC_c001_v001_r001_reg004"],
        k=3,
        microe_neighbor_cutoff=200.0,
        subset_cells=False,
        pre_transform = biomarker_pretransform,
        transform = mask_transform
    )
    print(len(dataset))  # number of MicroE subgraphs
    subgraph_0 = dataset[0]  # a PyG Data object
    print(subgraph_0)