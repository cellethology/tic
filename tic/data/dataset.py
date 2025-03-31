"""
Module: tic.data.dataset

Defines PyG-based datasets (InMemoryDataset) for microenvironment graphs, as
well as utilities to load or generate microenvironment dataloaders.
"""

import os
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from typing import Any, Callable, List, Optional
import anndata

from tic.data.graph_feature import edge_attr_fn, edge_index_fn, node_feature_fn
from tic.data.microe import MicroE
from utils.dataload import process_region_to_anndata

RawToAnnDataFunc = Callable[[str, str], anndata.AnnData]


class MicroEDataset(InMemoryDataset):
    """
    A PyG InMemoryDataset that creates microenvironment-level Data objects
    from raw data. The raw data is first converted to an AnnData object
    (using a user-supplied function or the default process_region_to_anndata),
    and then a Tissue is instantiated via Tissue.from_anndata.
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
        Initialize MicroEDataset.

        Parameters
        ----------
        root : str
            Root directory, which should contain "Raw/" and "Cache/".
        region_ids : List[str]
            List of region/tissue IDs.
        k : int
            k-hop neighborhood for extracting MicroE subgraphs.
        transform : Optional[Callable]
            Optional PyG transform.
        pre_transform : Optional[Callable]
            Optional PyG pre_transform.
        microe_neighbor_cutoff : float
            Distance threshold for filtering neighbors in microenvironments.
        subset_cells : bool
            If True, sample a subset of cells for large tissues.
        center_cell_types : List[str]
            List of cell types to consider as center cells.
        raw_to_anndata_func : Optional[RawToAnnDataFunc]
            Function to convert raw data to an AnnData object.
        """
        self.root: str = root
        self.region_ids: List[str] = region_ids
        self.k: int = k
        self.microe_neighbor_cutoff: float = microe_neighbor_cutoff
        self.subset_cells: bool = subset_cells
        self.center_cell_types: List[str] = list(center_cell_types)
        self.raw_to_anndata_func: Optional[RawToAnnDataFunc] = raw_to_anndata_func

        super().__init__(root, transform, pre_transform)
        self.data, self.slices, self.index_map = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        """Return the raw directory path."""
        raw_dir_path = os.path.join(self.root, "Raw")
        if not os.path.exists(raw_dir_path):
            raise FileNotFoundError(f"Raw data directory not found at {raw_dir_path}")
        return raw_dir_path

    @property
    def processed_dir(self) -> str:
        """Return the processed directory path, creating it if necessary."""
        processed_dir_path = os.path.join(self.root, "Cache")
        if not os.path.exists(processed_dir_path):
            os.makedirs(processed_dir_path)
        return processed_dir_path

    @property
    def raw_file_names(self) -> List[str]:
        """List expected raw filenames for each region."""
        files: List[str] = []
        for rid in self.region_ids:
            files.extend(
                [
                    f"{rid}.cell_data.csv",
                    f"{rid}.cell_features.csv",
                    f"{rid}.cell_types.csv",
                    f"{rid}.expression.csv",
                ]
            )
        return files

    @property
    def processed_file_names(self) -> List[str]:
        """Name of the processed dataset file."""
        fname = (
            f"microe_dataset_{len(self.region_ids)}_k{self.k}_"
            f"cutoff{self.microe_neighbor_cutoff}.pt"
        )
        return [fname]

    def download(self) -> None:
        """
        Download is not implemented; raw data is assumed to exist
        in the appropriate directory.
        """
        pass

    def process(self) -> None:
        """
        Process raw data to generate microenvironment subgraphs.

        For each region, use the provided raw_to_anndata_func (or the default
        process_region_to_anndata) to convert the raw files to an AnnData
        object, then instantiate a Tissue via Tissue.from_anndata.
        """
        data_list: List[Data] = []

        from tic.data.tissue import Tissue

        for rid in self.region_ids:
            print(f"[MicroEDataset] Processing Tissue {rid} ...")
            tissue_cache_path = os.path.join(self.processed_dir, f"Tissue_{rid}.pt")

            if os.path.exists(tissue_cache_path):
                tissue = torch.load(tissue_cache_path)
            else:
                if self.raw_to_anndata_func is not None:
                    adata = self.raw_to_anndata_func(self.raw_dir, rid)
                else:
                    adata = process_region_to_anndata(self.raw_dir, rid)

                tissue = Tissue.from_anndata(adata, tissue_id=rid)
                tissue.to_graph(node_feature_fn, edge_index_fn, edge_attr_fn)
                torch.save(tissue, tissue_cache_path)

            cell_list: List[Any] = [
                c for c in tissue.cells if c.cell_type in self.center_cell_types
            ]
            if self.subset_cells:
                cell_list = np.random.choice(
                    cell_list,
                    size=min(100, len(cell_list)),
                    replace=False
                ).tolist()

            for cell in cell_list:
                center_id: str = cell.cell_id
                micro_env: MicroE = tissue.get_microenvironment(
                    center_id,
                    k=self.k,
                    microe_neighbor_cutoff=self.microe_neighbor_cutoff
                )
                micro_graph: Data = micro_env.graph

                raw_microe_path = os.path.join(
                    self.processed_dir, f"MicroE_{rid}_{center_id}.pt"
                )
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
        """Return the number of data points in the dataset."""
        return self.slices["x"].size(0) - 1

    def get_microe_item(self, idx: int) -> MicroE:
        """
        Retrieve a single MicroE object by index.

        Parameters
        ----------
        idx : int
            Index in the dataset.

        Returns
        -------
        MicroE
            The corresponding microenvironment object.
        """
        rid, cid = self.index_map[idx]
        return self.get_microE(rid, cid)

    def get_microE(self, region_id: str, cell_id: str) -> MicroE:
        """
        Retrieve a stored MicroE object by region_id and cell_id.

        Parameters
        ----------
        region_id : str
            The tissue/region identifier.
        cell_id : str
            The center cell ID.

        Returns
        -------
        MicroE
            The stored MicroE object.
        """
        raw_microe_path = os.path.join(
            self.processed_dir, f"MicroE_{region_id}_{cell_id}.pt"
        )
        return torch.load(raw_microe_path)

    def get_Tissue(self, region_id: str) -> Any:
        """
        Retrieve a stored Tissue object for a given region.

        Parameters
        ----------
        region_id : str
            The tissue/region identifier.

        Returns
        -------
        Tissue
            The stored Tissue object.
        """
        tissue_cache_path = os.path.join(self.processed_dir, f"Tissue_{region_id}.pt")
        return torch.load(tissue_cache_path)


class MicroEWrapperDataset(torch.utils.data.Dataset):
    """
    A wrapper around MicroEDataset that yields MicroE objects.
    """

    def __init__(self, microe_dataset: MicroEDataset) -> None:
        """
        Initialize with a MicroEDataset.

        Parameters
        ----------
        microe_dataset : MicroEDataset
            The MicroEDataset to wrap.
        """
        self.mdataset: MicroEDataset = microe_dataset

    def __len__(self) -> int:
        """Return the length of the wrapped dataset."""
        return len(self.mdataset)

    def __getitem__(self, idx: int) -> MicroE:
        """Return a MicroE object by index."""
        return self.mdataset.get_microe_item(idx)


def create_microe_dataloader(
    mdataset: MicroEDataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader that yields MicroE objects.

    Parameters
    ----------
    mdataset : MicroEDataset
        The dataset of microenvironments.
    batch_size : int
        Number of microenvironments per batch.
    shuffle : bool
        Whether to shuffle the dataset each epoch.
    num_workers : int
        How many subprocesses to use for data loading.

    Returns
    -------
    torch.utils.data.DataLoader
        A DataLoader that yields batches of MicroE objects.
    """
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
    """
    Collate function for MicroE objects (simply returns a list).

    Parameters
    ----------
    batch : List[MicroE]
        A list of MicroE objects.

    Returns
    -------
    List[MicroE]
        The same list of MicroE objects.
    """
    return batch