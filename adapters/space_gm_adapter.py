"""
Space-gm adapter & extended classes
Athor: Jiahao Zhang
Time: 15:52 30 Dec 2024
"""
from concurrent.futures import ProcessPoolExecutor
import csv
import os
import pickle
import random
import numpy as np
import multiprocessing

import pandas as pd
from spacegm.data import InfDataLoader
import torch
from torch.utils.data import RandomSampler

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

#----------------------------------
# Data Class
#----------------------------------
#--------------------------------------------------
# Jiahao Zhang , Dec 18, 10:50, 2024
#--------------------------------------------------


class MemorySubgraphDataset(Dataset):
    """A simple in-memory dataset that wraps a list of pyg Data objects."""
    def __init__(self, subgraphs):
        super(MemorySubgraphDataset, self).__init__()
        self.subgraphs = subgraphs

    def __len__(self):
        return len(self.subgraphs)

    def __getitem__(self, idx):
        return self.subgraphs[idx]

class CellTypeSpecificSubgraphSampler(object):
    """
    Iterator for sampling subgraphs from a CellularGraphDataset where
    the center node is of a specified cell type.

    This sampler:
    - Takes a list of region indices (selected_inds) from the dataset.
    - For each segment, selects a subset of regions.
    - Loads all subgraphs (if subgraph_source='chunk_save') or computes them on-the-fly.
    - Filters these subgraphs so that only those whose center node's cell type matches the target cell_type remain.
    - Uses a DataLoader with a RandomSampler for infinite sampling of these filtered subgraphs.
    """

    def __init__(self,
                 dataset,
                 cell_type,
                 selected_inds=None,
                 batch_size=64,
                 num_regions_per_segment=32,
                 steps_per_segment=1000,
                 num_workers=0,
                 seed=None):
        """
        Args:
            dataset (CellularGraphDataset): The dataset instance.
            cell_type (int): The target cell type index for the center node.
            selected_inds (list): List of region indices to sample from.
            batch_size (int): Batch size for the DataLoader.
            num_regions_per_segment (int): How many regions to load each segment.
            steps_per_segment (int): How many batches to yield per segment before selecting new regions.
            num_workers (int): Number of workers for DataLoader.
            seed (int): Random seed for reproducibility.
        """
        self.dataset = dataset
        self.cell_type = cell_type
        self.selected_inds = list(dataset.indices()) if selected_inds is None else list(selected_inds)
        self.batch_size = batch_size
        self.num_regions_per_segment = num_regions_per_segment
        self.steps_per_segment = steps_per_segment
        self.num_workers = multiprocessing.cpu_count() if num_workers is None else num_workers

        self.region_inds_queue = []
        self.fill_queue(seed=seed)

        self.step_counter = 0
        self.data_iter = None
        self.get_new_segment()

    def fill_queue(self, seed=None):
        """Fill the queue of region indices randomly."""
        if seed is not None:
            np.random.seed(seed)
        fill_inds = sorted(set(self.selected_inds) - set(self.region_inds_queue))
        np.random.shuffle(fill_inds)
        self.region_inds_queue.extend(fill_inds)

    def set_subset_inds(self, selected_inds):
        """Set the dataset to only contain the given subset of region indices."""
        self.dataset.set_indices(selected_inds)

    def get_new_segment(self):
        """
        Load a new segment of regions, filter subgraphs by cell_type, 
        and prepare a DataLoader for infinite sampling.
        """
        if self.num_regions_per_segment <= 0:
            # Use all selected_inds
            self.set_subset_inds(self.selected_inds)
        else:
            # Sample a subset of regions from the queue
            graph_inds_in_segment = self.region_inds_queue[:self.num_regions_per_segment]
            self.region_inds_queue = self.region_inds_queue[self.num_regions_per_segment:]
            if len(self.region_inds_queue) < self.num_regions_per_segment:
                self.fill_queue()

            # Clear cache and set indices for these regions
            self.dataset.clear_cache()
            self.set_subset_inds(graph_inds_in_segment)

            # Pre-load graphs and their subgraphs if chunk_save is used
            try:
                for idx in self.dataset.indices():
                    self.dataset.load_to_cache(idx, subgraphs=True)
            except FileNotFoundError as e:
                print("Cannot find subgraph chunk save files, " +
                      "try running `dataset.save_all_subgraphs_to_chunk()` first")
                raise e

        # Filter subgraphs for the specified cell type
        filtered_subgraphs = []
        for idx in self.dataset.indices():
            full_data = self.dataset.get_full(idx)
            num_nodes = full_data.num_nodes
            # Find nodes of target cell type
            center_candidates = [n for n in range(num_nodes) if full_data.x[n, 0].item() == self.cell_type]

            # check
            for c_node in center_candidates:
                node_ct = full_data.x[c_node, 0].item()
                if node_ct != self.cell_type:
                    print(f"Warning: Center node {c_node} has cell_type {node_ct}, expected {self.cell_type}")
                    
            # Get subgraphs for each candidate
            for center_node in center_candidates:
                subg = self.dataset.get_subgraph(idx, center_node)
                filtered_subgraphs.append(subg)

        # Create a DataLoader from filtered subgraphs
        # If no subgraphs found, use empty list to avoid errors
        if len(filtered_subgraphs) == 0:
            # No subgraphs match the cell type
            data_source = MemorySubgraphDataset([])
        else:
            data_source = MemorySubgraphDataset(filtered_subgraphs)

        sampler = RandomSampler(data_source, replacement=True, num_samples=int(1e10))
        loader = InfDataLoader(data_source, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers)
        self.data_iter = iter(loader)
        self.step_counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.step_counter == self.steps_per_segment:
            self.get_new_segment()
        batch = next(self.data_iter, None)
        if batch is None:
            # If no subgraphs were found at all, stop iteration
            raise StopIteration
        self.step_counter += 1
        return batch

class FullCellTypeSubgraphIterator:
    """
    A complete iterator that traverses all subgraphs with a specified cell type as the center node.
    """

    def __init__(self, dataset, cell_type, batch_size=64, num_workers=0, output_csv="center_candidates.csv"):
        """
        Initialize the iterator.

        Args:
            dataset (CellularGraphDataset): The dataset instance.
            cell_type (int): The target cell type for the center node.
            batch_size (int): The batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
            output_csv (str): Path to save the CSV file with center_candidates info.
        """
        self.dataset = dataset
        self.cell_type = cell_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.all_subgraphs = []
        self.output_csv = output_csv
        self.center_candidates_info = []  # To store region_id and center node info

        self._prepare_subgraphs()
        self._save_to_csv()

    def _prepare_subgraphs(self):
        """
        Prepare all subgraphs with the specified cell type as the center node.
        """
        seen = set()  # deal with seen (region_id, center_node_idx) pairs
        
        # Iterate over all regions in the dataset
        for idx in range(len(self.dataset)):
            full_graph = self.dataset.get_full(idx)
            region_id = full_graph.region_id  # Extract the actual region_id from the graph

            # Identify center nodes matching the target cell type
            center_candidates = (full_graph.x[:, 0] == self.cell_type).nonzero(as_tuple=True)[0].tolist()

            # Generate subgraphs for each candidate and record their information
            for center_node in center_candidates:
                if (region_id, center_node) in seen:
                    continue  
                
                subgraph = self.dataset.get_subgraph(idx, center_node)
                self.all_subgraphs.append(subgraph)
                self.center_candidates_info.append({"region_id": region_id, "center_node_idx": center_node})
                seen.add((region_id, center_node))  


    def _save_to_csv(self):
        """
        Save center_candidates information to a CSV file.
        """
        with open(self.output_csv, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["region_id", "center_node_idx"])
            writer.writeheader()
            writer.writerows(self.center_candidates_info)

        print(f"Center candidates information saved to {self.output_csv}")

    def get_dataloader(self):
        """
        Create a DataLoader for traversing the prepared subgraphs.

        Returns:
            DataLoader: A PyG DataLoader for batch processing the subgraphs.
        """
        # Use MemorySubgraphDataset for efficient in-memory batching
        subgraph_dataset = MemorySubgraphDataset(self.all_subgraphs)
        print(f"Total Num of Subgraphs with Cell Type:{self.cell_type} is {len(subgraph_dataset)}")
        return DataLoader(subgraph_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class CustomSubgraphSampler:
    """
    Custom Sampler for sampling subgraphs based on the specified conditions.

    Supports sampling across all regions or specific regions, and across different
    or specific cell types.

    Args:
        dataset (CellularGraphDataset): The dataset instance.
        total_samples (int): The total number of samples to retrieve.
        cell_type (int or list, optional): Specific cell type(s) to sample. Defaults to None.
        region_id (str, optional): Specific region ID to sample. Defaults to None.
        batch_size (int, optional): Batch size for DataLoader. Defaults to None.
        num_workers (int): Number of workers for DataLoader.
        output_csv (str, optional): Path to save the CSV file with sampled subgraph info.
        include_node_info (bool): Whether to include all node-level information in the sampled info.
        random_seed (int, optional): Random seed for reproducibility.
    """

    def __init__(self, 
                 dataset, 
                 total_samples=1000, 
                 cell_type=None, 
                 region_id=None, 
                 batch_size=None, 
                 num_workers=0, 
                 output_csv="sampled_subgraphs.csv",
                 include_node_info=False,
                 random_seed=None):
        self.dataset = dataset
        self.total_samples = total_samples
        self.cell_type = cell_type if isinstance(cell_type, list) and not any(isinstance(ct, list) for ct in cell_type) else [
            item for sublist in cell_type for item in (sublist if isinstance(sublist, list) else [sublist])]
        self.region_id = region_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.output_csv = output_csv
        self.include_node_info = include_node_info
        self.sampled_subgraphs = []

        output_dir = os.path.dirname(output_csv)
        os.makedirs(output_dir, exist_ok=True)

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            try:
                import torch
                torch.manual_seed(random_seed)
            except ImportError:
                pass

        self._prepare_samples_proportionally()
        self._save_to_csv()

    def _process_region(self, region_idx, num_samples, cell_types, include_node_info, seen):
        """Helper function to process a single region."""
        sampled_subgraphs = []
        full_graph = self.dataset.get_full(region_idx)
        region_id = full_graph.region_id

        # Retrieve candidates for specified cell types
        center_candidates = []
        for cell_type in cell_types:
            type_candidates = (full_graph.x[:, 0] == cell_type).nonzero(as_tuple=True)[0].tolist()
            center_candidates.extend(type_candidates)

        # Ensure num_samples does not exceed the number of candidates
        if len(center_candidates) == 0:
            print(f"Warning: No candidates found for region {region_id} and cell types {cell_types}. Skipping.")
            return sampled_subgraphs

        actual_num_samples = min(num_samples, len(center_candidates))
        sampled_nodes = np.random.choice(center_candidates, actual_num_samples, replace=False)

        for center_node in sampled_nodes:
            if (region_id, center_node) in seen:
                continue

            subgraph = self.dataset.get_subgraph(region_idx, center_node)
            cell_id = None
            try:
                cell_id = self._get_cell_id_from_graph(region_idx, center_node)
            except Exception as e:
                print(f"Error retrieving cell ID for region {region_id}, node {center_node}: {e}")

            # Determine the cell type of the center node
            center_cell_type = int(full_graph.x[center_node, 0].item())

            # Include optional node info
            node_info = None
            if include_node_info:
                try:
                    node_info = self._get_node_info(region_idx, center_node)
                except Exception as e:
                    print(f"Error retrieving node info for region {region_id}, node {center_node}: {e}")

            sampled_subgraphs.append({
                "region_id": region_id,
                "center_node_idx": center_node,
                "cell_id": cell_id,
                "cell_type": center_cell_type,  
                "subgraph": subgraph,
                "node_info": node_info,
            })
        return sampled_subgraphs

    def _prepare_samples_proportionally(self):
        """Optimized preparation of subgraphs using multiprocessing."""
        # Preload region data
        region_subgraph_counts = [self.dataset.get_full(idx).num_nodes for idx in range(len(self.dataset))]
        total_subgraphs = sum(region_subgraph_counts)

        print(f'Total Subgraphs: {total_subgraphs}')
        region_weights = np.array(region_subgraph_counts) / total_subgraphs
        region_allocations = (self.total_samples * region_weights).astype(int)

        # Ensure total_samples is accurate
        adjustment = self.total_samples - region_allocations.sum()
        if adjustment > 0:
            region_allocations[:adjustment] += 1

        seen = set()

        # Use multiprocessing to process regions
        sampled_subgraphs = []
        with ProcessPoolExecutor() as executor:
            tasks = [
                executor.submit(
                    self._process_region, 
                    idx, num_samples, self.cell_type, self.include_node_info, seen
                )
                for idx, num_samples in enumerate(region_allocations)
            ]
            for task in tasks:
                sampled_subgraphs.extend(task.result())

        self.sampled_subgraphs = sampled_subgraphs

    def _get_cell_id_from_graph(self, region_idx, center_node_idx):
        """Retrieve the cell ID for a center node from the raw graph."""
        graph_path = os.path.join(self.dataset.raw_dir, f"{self.dataset.region_ids[region_idx]}.gpkl")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Raw graph file not found for region: {self.dataset.region_ids[region_idx]}")

        nx_graph = pickle.load(open(graph_path, "rb"))
        for node, data in nx_graph.nodes(data=True):
            if node == center_node_idx:
                return data.get("cell_id", None)
        raise ValueError(f"Cell ID for center node {center_node_idx} not found in region {self.dataset.region_ids[region_idx]}.")

    def _get_node_info(self, region_idx, center_node_idx):
        """Retrieve detailed information for a specific node."""
        graph_path = os.path.join(self.dataset.raw_dir, f"{self.dataset.region_ids[region_idx]}.gpkl")
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Raw graph file not found for region: {self.dataset.region_ids[region_idx]}")

        nx_graph = pickle.load(open(graph_path, "rb"))
        if center_node_idx in nx_graph.nodes:
            return nx_graph.nodes[center_node_idx]
        else:
            raise ValueError(f"Node {center_node_idx} not found in region {self.dataset.region_ids[region_idx]}.")

    def _save_to_csv(self):
        """Save sampled subgraphs information to a CSV file."""
        with open(self.output_csv, mode="w", newline="") as file:
            fieldnames = ["region_id", "center_node_idx", "cell_id", "cell_type"]  
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for subgraph_info in self.sampled_subgraphs:
                row = {
                    "region_id": subgraph_info["region_id"],
                    "center_node_idx": subgraph_info["center_node_idx"],
                    "cell_id": subgraph_info["cell_id"],
                    "cell_type": subgraph_info["cell_type"],  
                }
                writer.writerow(row)
        print(f"Sampled subgraphs information saved to {self.output_csv}")

    def get_output_as_dataframe(self):
        """Return the output CSV as a pandas DataFrame."""
        if not os.path.exists(self.output_csv):
            raise FileNotFoundError(f"Output CSV file {self.output_csv} does not exist.")
        return pd.read_csv(self.output_csv)

    def get_all_sampled_subgraphs(self):
        """Return all sampled subgraphs as a list."""
        return self.sampled_subgraphs

    def get_subgraph_objects(self):
        """Return a list of subgraphs as PyG data objects."""
        return [s["subgraph"] for s in self.sampled_subgraphs]

    def get_dataloader(self):
        """
        Optionally create a DataLoader for traversing the sampled subgraphs if batch_size is specified.

        Returns:
            DataLoader: A PyG DataLoader for batch processing the subgraphs.
        """
        if self.batch_size is None:
            raise ValueError("Batch size not specified. Cannot create a DataLoader.")

        subgraph_dataset = MemorySubgraphDataset(self.get_subgraph_objects())
        print(f"Total Sampled Subgraphs: {len(subgraph_dataset)}")
        return DataLoader(subgraph_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def add_kv_to_sampled_subgraphs(self, values, key="embeddings"):
        """
        Add values to each sampled subgraph, ensuring alignment.

        Args:
            values (list): List of values to align with self.sampled_subgraphs.
            key (str): Key under which embeddings are stored in sampled_subgraphs.
        """
        if len(values) != len(self.sampled_subgraphs):
            raise ValueError("value length must match the number of sampled subgraphs.")

        for subgraph, value in zip(self.sampled_subgraphs, values):
            subgraph[key] = value

    
def find_subgraph_by_region_and_node(dataset, region_id, center_node_idx):
    """
    Retrieve the subgraph for a specific region_id and center_node_idx.

    Args:
        dataset (CellularGraphDataset): The dataset instance.
        region_id (str): The region ID to search for.
        center_node_idx (int): The center node index.

    Returns:
        torch_geometric.data.Data: The corresponding subgraph.
    """
    # Step 1: Find the region index for the given region_id
    try:
        region_idx = dataset.region_ids.index(region_id)
    except ValueError:
        raise ValueError(f"Region ID {region_id} not found in the dataset.")

    # Step 2: Retrieve the subgraph using the region index and center node index
    try:
        subgraph = dataset.get_subgraph(region_idx, center_node_idx)
    except KeyError:
        raise KeyError(f"Subgraph for region {region_id} and center node {center_node_idx} not found.")

    return subgraph

#----------------------------------
# Helper Functions 
#----------------------------------
#--------------------------------------------------
# Jiahao Zhang , Dec 18, 10:44, 2024
#--------------------------------------------------
def sample_subgraphs_by_cell_type(dataset, cell_type, inds=None, n_samples=32768, batch_size=64,
                                  num_workers=0, seed=123):
    """
    Sample subgraphs with the specified `cell_type` as the center node.

    Args:
        dataset (CellularGraphDataset): Target dataset.
        cell_type (int): Target center node `cell_type` (integer index).
        n_samples (int): Number of subgraphs to sample.
        batch_size (int): Batch size for sampling.
        num_regions_per_segment (int): Number of regions to sample from in each segment.
        steps_per_segment (int): Number of batches in each segment.
        num_workers (int): Number of workers for multiprocessing.
        seed (int): Random seed for reproducibility.

    Returns:
        list: List of subgraphs (as PyG data objects) with the specified `cell_type` as the center node.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    original_indices = dataset.indices()
    # Reset dataset indices
    dataset.set_indices()
    if inds is None:
        inds = np.arange(dataset.N)
    n_iterations = int(np.ceil(n_samples / batch_size))
    # Set up the sampler
    data_iter = CellTypeSpecificSubgraphSampler(dataset,
                                cell_type=cell_type,
                                selected_inds=inds,
                                batch_size=batch_size,
                                num_regions_per_segment=0,
                                steps_per_segment=n_samples + 1,
                                num_workers=num_workers,
                                seed=seed)
    # Sample subgraphs
    data_list = []
    for _ in range(n_iterations):
        batch = next(data_iter)
        data_list.extend(batch.to_data_list())
    dataset.set_indices(original_indices)
    return data_list[:n_samples]

def get_all_subgraphs_by_cell_type(dataset,
                                  cell_type,
                                  batch_size=64,
                                  num_workers=0):
    """
    Sample subgraphs where the center node is of the specified cell_type.

    Args:
        dataset (CellularGraphDataset): The graph dataset.
        cell_type (int): The target cell type.
        batch_size (int): Batch size for processing subgraphs.
        num_workers (int): Number of workers for loading data.

    Returns:
        DataLoader: A PyG DataLoader containing subgraphs with the specified center node.
    """
    all_center_candidates = []

    # Step 1: Find all valid center nodes for the specified cell type
    for idx in range(len(dataset)):
        data = dataset.get_full(idx)  # Get the full graph
        # Find indices where the cell_type matches
        center_candidates = (data.x[:, 0] == cell_type).nonzero(as_tuple=True)[0]
        center_candidates = center_candidates.cpu().numpy()
        all_center_candidates.extend([(idx, center) for center in center_candidates])

    # Step 2: Create subgraphs for all valid center nodes
    subgraph_list = []
    for idx, center in all_center_candidates:
        subgraph = dataset.get_subgraph(idx, center)
        subgraph_list.append(subgraph)

    # Step 3: Wrap the subgraphs into a DataLoader for batch processing
    dataloader = DataLoader(subgraph_list, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader