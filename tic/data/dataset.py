# tic/data/dataset.py
import os
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from typing import Any, Callable, List, Optional, Tuple
import anndata

from tic.data.graph_feature import edge_attr_fn, edge_index_fn, node_feature_fn
from tic.data.microe import MicroE
from utils.dataload import process_region_to_anndata

RawToAnnDataFunc = Callable[[str, str], anndata.AnnData]

class MicroEDataset(InMemoryDataset):
    """
    A PyG InMemoryDataset that creates microenvironment-level Data objects from raw data.
    The raw data is first converted to an AnnData object (using a user-supplied function
    or the default process_region_to_anndata), and then a Tissue is instantiated via Tissue.from_anndata.
    """
    def __init__(
        self,
        root: str,
        region_ids: List[str],
        k: int = 3,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        microe_neighbor_cutoff: float = 200.0,
        subset_cells: bool = False,
        center_cell_types: List[str] = ["Tumor"],
        raw_to_anndata_func: Optional[RawToAnnDataFunc] = None  
    ) -> None:
        """
        :param root: Root directory; should contain 'Raw/' and 'Cache/' subfolders.
        :param region_ids: List of region/tissue IDs.
        :param k: k-hop neighborhood for extracting MicroE subgraphs.
        :param transform: Optional PyG transform.
        :param pre_transform: Optional PyG pre_transform.
        :param microe_neighbor_cutoff: Distance threshold for filtering neighbors.
        :param subset_cells: If True, sample a subset of cells for large tissues.
        :param center_cell_types: List of cell types to consider as center cells.
        :param raw_to_anndata_func: Optional function to convert raw data to AnnData.
        """
        self.root: str = root
        self.region_ids: List[str] = region_ids
        self.k: int = k
        self.microe_neighbor_cutoff: float = microe_neighbor_cutoff
        self.subset_cells: bool = subset_cells
        self.center_cell_types: List[str] = center_cell_types
        self.raw_to_anndata_func: Optional[RawToAnnDataFunc] = raw_to_anndata_func
        super().__init__(root, transform, pre_transform)
        
        # Load the processed dataset.
        self.data, self.slices, self.index_map = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        raw_dir_path = os.path.join(self.root, "Raw")
        if not os.path.exists(raw_dir_path):
            raise FileNotFoundError(f"Raw data directory not found at {raw_dir_path}")
        return raw_dir_path

    @property
    def processed_dir(self) -> str:
        processed_dir_path = os.path.join(self.root, "Cache")
        if not os.path.exists(processed_dir_path):
            os.makedirs(processed_dir_path)
        return processed_dir_path

    @property
    def raw_file_names(self) -> List[str]:
        files: List[str] = []
        for rid in self.region_ids:
            files.extend([
                f"{rid}.cell_data.csv",
                f"{rid}.cell_features.csv",
                f"{rid}.cell_types.csv",
                f"{rid}.expression.csv"
            ])
        return files

    @property
    def processed_file_names(self) -> List[str]:
        fname = f"microe_dataset_{len(self.region_ids)}_k{self.k}_cutoff{self.microe_neighbor_cutoff}.pt"
        return [fname]

    def download(self) -> None:
        pass

    def process(self) -> None:
        """
        Process raw data to generate microenvironment subgraphs.
        For each region, use the provided raw_to_anndata_func (or the default process_region_to_anndata)
        to convert the raw files to an AnnData object, and then instantiate a Tissue via Tissue.from_anndata.
        """
        data_list: List[Data] = []
        self.index_map: List[Tuple[str, str]] = []

        for rid in self.region_ids:
            print(f"[MicroEDataset] Processing Tissue {rid} ...")
            tissue_cache_path = os.path.join(self.processed_dir, f"Tissue_{rid}.pt")
            if os.path.exists(tissue_cache_path):
                tissue = torch.load(tissue_cache_path)
            else:
                if self.raw_to_anndata_func is not None:
                    adata = self.raw_to_anndata_func(self.raw_dir, rid)
                else:
                    # Use the default function.
                    adata = process_region_to_anndata(self.raw_dir, rid)
                # Instantiate Tissue from AnnData.
                from tic.data.tissue import Tissue  # local import to avoid circular dependency.
                tissue = Tissue.from_anndata(adata, tissue_id=rid, position=None)
                tissue.to_graph(node_feature_fn, edge_index_fn, edge_attr_fn)
                torch.save(tissue, tissue_cache_path)

            cell_list: List[Any] = [c for c in tissue.cells if c.cell_type in self.center_cell_types]
            if self.subset_cells:
                cell_list = np.random.choice(cell_list, size=min(100, len(cell_list)), replace=False).tolist()

            for cell in cell_list:
                center_id: str = cell.cell_id
                micro_env: MicroE = tissue.get_microenvironment(
                    center_id,
                    k=self.k,
                    microe_neighbor_cutoff=self.microe_neighbor_cutoff
                )
                micro_graph: Data = micro_env.graph

                raw_microe_path = os.path.join(self.processed_dir, f"MicroE_{rid}_{center_id}.pt")
                torch.save(micro_env, raw_microe_path)

                if self.pre_transform is not None:
                    micro_graph = self.pre_transform(micro_graph)

                setattr(micro_graph, "region_id", rid)
                setattr(micro_graph, "cell_id", center_id)

                data_list.append(micro_graph)
                self.index_map.append((rid, center_id))

        data, slices = self.collate(data_list)
        torch.save((data, slices, self.index_map), self.processed_paths[0])

    def len(self) -> int:
        return self.slices['x'].size(0) - 1

    def get_microe_item(self, idx: int) -> MicroE:
        rid, cid = self.index_map[idx]
        return self.get_microE(rid, cid)
    
    def get_microE(self, region_id: str, cell_id: str) -> MicroE:
        raw_microe_path = os.path.join(self.processed_dir, f"MicroE_{region_id}_{cell_id}.pt")
        return torch.load(raw_microe_path)
    
    def get_Tissue(self, region_id: str) -> Any:
        tissue_cache_path = os.path.join(self.processed_dir, f"Tissue_{region_id}.pt")
        return torch.load(tissue_cache_path)


class MicroEWrapperDataset(torch.utils.data.Dataset):
    """
    A wrapper around MicroEDataset that yields MicroE objects.
    """
    def __init__(self, microe_dataset: MicroEDataset) -> None:
        self.mdataset: MicroEDataset = microe_dataset
    
    def __len__(self) -> int:
        return len(self.mdataset)
    
    def __getitem__(self, idx: int) -> MicroE:
        return self.mdataset.get_microe_item(idx)


def create_microe_dataloader(
    mdataset: MicroEDataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    wrapper_ds = MicroEWrapperDataset(mdataset)
    loader = torch.utils.data.DataLoader(
        wrapper_ds, 
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_microe
    )
    return loader


def collate_microe(batch: List[MicroE]) -> List[MicroE]:
    return batch