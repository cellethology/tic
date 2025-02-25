import os
import random
from typing import Counter
import numpy as np
import pickle
import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph
from adapters.feature import process_biomarker_expression
from adapters.transform import mask_biomarker_expression
from adapters.utils import get_cell_ids_by_type
from spacegm.utils import BIOMARKERS_UPMC, CELL_TYPE_MAPPING_UPMC

class TumorCellGraphDataset(Dataset):
    """
    A dataset class for processing and managing tumor cell graph data.
    
    Args:
        dataset_root (str): The root directory containing the dataset.
        node_features (list[str]): List of node features to be considered.
        biomarker_list (list): List of all biomarker types in the dataset.
        cell_type_mapping (dict): Mapping of cell types for one-hot encoding.
        edge_features (list, optional): List of edge features to be considered. Defaults to None.
        transform (callable, optional): A function/transform to apply to the data.
        pre_transform (callable, optional): A function/transform to apply to the data before saving.
    """

    def __init__(self, dataset_root, node_features=None, biomarker_list=None, 
                 cell_type_mapping=None, edge_features=None, transform=None, pre_transform=None):
        node_features = node_features or ['center_coord', 'SIZE', 'cell_type', 'biomarker_expression']
        biomarker_list = biomarker_list or BIOMARKERS_UPMC
        cell_type_mapping = cell_type_mapping or CELL_TYPE_MAPPING_UPMC
        self.dataset_root = dataset_root
        self.biomarker_list = biomarker_list
        self.cell_type_mapping = cell_type_mapping
        self.node_features = node_features
        self.edge_features = edge_features if edge_features else ['distance', 'edge_type']
        self.transform = transform
        self.pre_transform = pre_transform
        self.region_ids = [path.split('.')[0] for path in os.listdir(os.path.join(dataset_root, 'graph'))]
        self.feature_indices = self.describe_features()

        # Subgraph cache directory
        self.subgraph_cache_dir = os.path.join(self.dataset_root, 'subgraph')
        os.makedirs(self.subgraph_cache_dir, exist_ok=True)

    def process_graph(self, region_id):
        """
        Process a single region's data to create a graph and include cell_id as a feature.
        
        Args:
            region_id (str): The region ID for which to create the graph.
        
        Returns:
            Data: PyG Data object containing the region's graph information.
        """
        graph_file = os.path.join(self.dataset_root, 'graph', f'{region_id}.gpkl')
        with open(graph_file, 'rb') as f:
            G = pickle.load(f)

        # Extract node and edge features
        node_features = self.extract_node_features(G)
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        edge_attr = self.extract_edge_features(G)

        # Add additional node features (e.g., cell_id and position)
        cell_ids = torch.tensor([data['cell_id'] for _, data in G.nodes(data=True)], dtype=torch.long)
        cell_id_to_node_idx = {data['cell_id']: idx for idx, (node, data) in enumerate(G.nodes(data=True))}
        pos = torch.tensor([data['center_coord'] for _, data in G.nodes(data=True)], dtype=torch.float32)

        # Create Data object for PyG
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, cell_id=cell_ids, pos=pos)
        data.cell_id_to_node_idx = cell_id_to_node_idx
        data.feature_indices = self.feature_indices

        # Apply pre-transform if exists
        if self.pre_transform:
            data = self.pre_transform(data)

        return data

    def describe_features(self):
        """
        Describes the feature indices for each feature type in the dataset.
        
        Returns:
            dict: Dictionary containing feature names and their corresponding indices.
        """
        feature_indices = {}
        total_dim = 0

        # Describing node features based on feature selection
        if 'center_coord' in self.node_features:
            feature_indices['center_coord'] = (total_dim, total_dim + 2)
            total_dim += 2
        
        if 'biomarker_expression' in self.node_features:
            biomarker_start = total_dim
            biomarker_end = total_dim + len(self.biomarker_list)
            feature_indices['biomarker_expression'] = (biomarker_start, biomarker_end)
            total_dim += len(self.biomarker_list)
        
        if 'SIZE' in self.node_features:
            feature_indices['SIZE'] = (total_dim, total_dim + 1)
            total_dim += 1
        
        if 'cell_type' in self.node_features:
            cell_type_start = total_dim
            cell_type_end = total_dim + len(self.cell_type_mapping.keys())
            feature_indices['cell_type'] = (cell_type_start, cell_type_end)
            total_dim += len(self.cell_type_mapping.keys())
            
        feature_indices['total_dim'] = total_dim
        return feature_indices
    
    def extract_node_features(self, G):
        """
        Extract node features based on the provided node feature list.
        
        Args:
            G (nx.Graph): The graph object containing the node features.
        
        Returns:
            torch.Tensor: Node features as a tensor.
        """
        node_features = []
        for node, data in G.nodes(data=True):
            feature_list = []

            # Add 'center_coord' feature
            if 'center_coord' in self.node_features:
                feature_list.extend(np.array(data['center_coord']).flatten())

            # Add 'biomarker_expression' feature
            if 'biomarker_expression' in self.node_features:
                biomarker_expr = np.array([data['biomarker_expression'].get(bm, 0) for bm in self.biomarker_list])
                biomarker_expr = process_biomarker_expression(biomarker_expr)  # Apply predefined processing function
                feature_list.extend(biomarker_expr)

            # Add 'SIZE' feature
            if 'SIZE' in self.node_features:
                feature_list.append(data['SIZE'])

            # Add 'cell_type' feature (one-hot encoding)
            if 'cell_type' in self.node_features:
                cell_type_index = self.cell_type_mapping.get(data['cell_type'], -1)
                cell_type_one_hot = np.zeros(len(self.cell_type_mapping))
                if cell_type_index != -1:
                    cell_type_one_hot[cell_type_index] = 1
                feature_list.extend(cell_type_one_hot)

            node_features.append(feature_list)

        return torch.tensor(node_features, dtype=torch.float32)

    def extract_edge_features(self, G):
        """
        Extract edge features based on edge attributes in the graph.
        
        Args:
            G (nx.Graph): The graph object containing the edge features.
        
        Returns:
            torch.Tensor: Edge features as a tensor.
        """
        edge_attr = []
        for _, _, data in G.edges(data=True):
            edge_feature = []

            # Add edge features
            if 'edge_type' in self.edge_features:
                edge_feature.append(1 if data['edge_type'] == 'neighbor' else 0)
            if 'distance' in self.edge_features:
                edge_feature.append(float(data['distance']))

            edge_attr.append(edge_feature)
        
        return torch.tensor(edge_attr, dtype=torch.float32)

    def get_subgraph(self, region_id, cell_id, k=3):
        """
        Retrieves a k-hop subgraph for a given cell_id in a specific region, with caching to disk.
        
        Args:
            region_id (str): The region ID.
            cell_id (int): The cell ID to use as the center node.
            k (int): The number of hops (default is 3).
        
        Returns:
            Data: PyG Data object representing the k-hop subgraph.
        """
        cache_filename = os.path.join(self.subgraph_cache_dir, f'{region_id}_{cell_id}_subgraph.pkl')

        # Check if subgraph is already cached
        if os.path.exists(cache_filename):
            with open(cache_filename, 'rb') as f:
                return pickle.load(f)

        data = self.process_graph(region_id)

        node_idx = data.cell_id_to_node_idx[cell_id]
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx, k, data.edge_index, relabel_nodes=True)

        # Filter node and edge features using the subset and edge_mask
        node_features = data.x[subset]
        edge_attr = data.edge_attr[edge_mask]
        pos = data.pos[subset]

        subgraph_data = Data(
            x=node_features, 
            edge_index=edge_index, 
            edge_attr=edge_attr,
            y=None, 
            pos=pos,
            global_node_idx=torch.tensor([node_idx]),
            relabeled_node_idx=mapping,
            feature_indices=self.feature_indices
        )

        # Apply transform if exists
        if self.transform:
            subgraph_data = self.transform(subgraph_data)

        # Save the subgraph to cache
        with open(cache_filename, 'wb') as f:
            pickle.dump(subgraph_data, f)

        return subgraph_data

    def __getitem__(self, idx):
        region_id = self.region_ids[idx]
        return self.process_graph(region_id)

    def __len__(self):
        return len(self.region_ids)

    def clear_cache(self):
        """Clears the subgraph cache directory."""
        for filename in os.listdir(self.subgraph_cache_dir):
            file_path = os.path.join(self.subgraph_cache_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)


class NaiveSubgraphSampler(torch.utils.data.IterableDataset):
    """
    A naive subgraph sampler for efficiently sampling k-hop subgraphs from the dataset.
    
    Args:
        dataset (Dataset): The dataset containing graph data.
        k (int): The number of hops for the k-hop subgraph.
        batch_size (int): The number of subgraphs per batch.
        shuffle (bool): Whether to shuffle the data before sampling.
        drop_last (bool): Whether to drop the last incomplete batch.
        cell_types (list, optional): List of cell types to sample from. If None, all cell types are used.
        infinite (bool): Whether to keep sampling indefinitely.
    """
    def __init__(self, dataset, k=3, batch_size=32, shuffle=True, drop_last=False, 
                 cell_types=None, infinite=True):
        self.dataset = dataset
        self.k = k
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.cell_types = cell_types  # List of cell types to sample from
        self.infinite = infinite  # Flag for infinite iteration
        
        # Initialize region_ids and shuffle them if required
        self.region_ids = self.dataset.region_ids
        if self.shuffle:
            self._shuffle_region_ids()

    def _shuffle_region_ids(self):
        """Shuffles the region IDs randomly."""
        shuffled_indices = torch.randperm(len(self.region_ids)).tolist()
        self.region_ids = [self.region_ids[i] for i in shuffled_indices]

    def __len__(self):
        """
        Returns the number of batches in the dataset (if not infinite).
        """
        if self.infinite:
            return float('inf')  # Infinite length for continuous sampling
        else:
            return len(self.region_ids) // self.batch_size

    def __iter__(self):
        """
        Infinite iteration over the dataset to return batches of subgraphs.
        """
        idx = 0
        while True:  # Endless loop for uninterrupted sampling
            if idx + self.batch_size > len(self.region_ids):
                if not self.drop_last:
                    remaining_ids = self.region_ids[idx:]
                    self.region_ids = self.region_ids[:idx]
                    self.region_ids.extend(remaining_ids)
                    idx = 0
                else:
                    break

            batch_region_ids = self.region_ids[idx:idx + self.batch_size]
            batch_cell_ids = self._get_random_cell_ids(batch_region_ids)

            subgraphs = [self._sample_subgraph(region_id, cell_id) 
                         for region_id, cell_id in zip(batch_region_ids, batch_cell_ids)]

            batch_data = Batch.from_data_list(subgraphs)
            yield batch_data

            idx += self.batch_size

    def _get_random_cell_ids(self, batch_region_ids):
        """
        Retrieves and randomly selects a cell_id for each region in the batch from raw data.
        
        Args:
            batch_region_ids (list): List of region IDs for the batch.
        
        Returns:
            List: A list of randomly selected cell IDs for the batch.
        """
        batch_cell_ids = []
        for region_id in batch_region_ids:
            # Fetch cell ids based on types from raw files
            cell_ids = self._get_cell_ids_by_type(region_id)  # Get cell_ids based on cell_types
            batch_cell_ids.append(random.choice(cell_ids))  # Randomly select a cell_id
        return batch_cell_ids

    def _get_cell_ids_by_type(self, region_id):
        """
        Retrieves cell IDs from the raw data files based on the specified cell types.

        Args:
            region_id (str): The region ID for which to get the cell IDs.
        
        Returns:
            list: List of cell IDs filtered by the specified cell types.
        """
        # Load the respective CSV files from the 'voronoi' folder
        cell_types_file = os.path.join(self.dataset.dataset_root, 'voronoi', f'{region_id}.cell_types.csv')        
        cell_types_df = pd.read_csv(cell_types_file)

        # If cell_types are specified, filter cell_ids by the selected types
        if self.cell_types:
            filtered_cell_types = cell_types_df[cell_types_df['CELL_TYPE'].isin(self.cell_types)]
            return filtered_cell_types['CELL_ID'].tolist()
        else:
            # If no filter is specified, return all cell IDs
            return cell_types_df['CELL_ID'].tolist()

    def _sample_subgraph(self, region_id, cell_id):
        """
        Samples a k-hop subgraph for a given region and cell_id.
        
        Args:
            region_id (str): The region ID.
            cell_id (int): The cell ID to use as the center node.
        
        Returns:
            Data: A PyG Data object representing the k-hop subgraph.
        """
        return self.dataset.get_subgraph(region_id, cell_id, self.k)            
import random
import torch
import pandas as pd
import os
from torch_geometric.data import Batch

class WeightedSubgraphSampler(NaiveSubgraphSampler):
    """
    A weighted subgraph sampler for efficiently sampling k-hop subgraphs from the dataset.
    This version caches cell IDs for each region and performs weighted sampling based on the number of valid cells in each region.

    Args:
        dataset (Dataset): The dataset containing graph data.
        k (int): The number of hops for the k-hop subgraph.
        batch_size (int): The number of subgraphs per batch.
        shuffle (bool): Whether to shuffle the data before sampling.
        drop_last (bool): Whether to drop the last incomplete batch.
        cell_types (list, optional): List of cell types to sample from. If None, all cell types are used.
        infinite (bool): Whether to keep sampling indefinitely.
    """
    
    def __init__(self, dataset, k=3, batch_size=32, shuffle=True, drop_last=False, 
                 cell_types=None, infinite=True):
        super().__init__(dataset, k, batch_size, shuffle, drop_last, cell_types, infinite)
        
        # Cache cell IDs per region for efficient lookups
        self.region_cell_id_map = self._cache_region_cell_ids()

    def _cache_region_cell_ids(self):
        """
        Caches cell ids for each region. This is done once to improve efficiency when sampling.
        
        Returns:
            dict: A dictionary mapping region_id to a list of valid cell IDs.
        """
        region_cell_id_map = {}
        for region_id in self.dataset.region_ids:
            cell_ids = self._get_cell_ids_by_type(region_id)
            region_cell_id_map[region_id] = cell_ids
        return region_cell_id_map

    def _get_valid_cell_count(self, region_id):
        """
        Returns the number of valid cells for a region based on cell type filtering.
        
        Args:
            region_id (str): The region ID for which to count valid cells.
        
        Returns:
            int: The number of valid cells in the region.
        """
        return len(self.region_cell_id_map[region_id])

    def _get_weighted_region_ids(self):
        """
        Samples region IDs with weights proportional to the number of valid cells.
        
        Returns:
            list: List of region IDs sampled with probability proportional to the number of valid cells.
        """
        weights = [self._get_valid_cell_count(region_id) for region_id in self.region_ids]
        total_weight = sum(weights)
        normalized_weights = [weight / total_weight for weight in weights]
        return random.choices(self.region_ids, weights=normalized_weights, k=len(self.region_ids))

    def __iter__(self):
        """
        Infinite iteration over the dataset to return batches of subgraphs with weighted region sampling.
        """
        idx = 0
        while True:  # Endless loop for uninterrupted sampling
            if idx + self.batch_size > len(self.region_ids):
                if not self.drop_last:
                    remaining_ids = self.region_ids[idx:]
                    self.region_ids = self.region_ids[:idx]
                    self.region_ids.extend(remaining_ids)
                    idx = 0
                else:
                    break

            # Sample regions based on weights
            batch_region_ids = self._get_weighted_region_ids()[idx:idx + self.batch_size]
            batch_cell_ids = self._get_random_cell_ids(batch_region_ids)

            subgraphs = [self._sample_subgraph(region_id, cell_id) 
                         for region_id, cell_id in zip(batch_region_ids, batch_cell_ids)]

            batch_data = Batch.from_data_list(subgraphs)
            yield batch_data

            idx += self.batch_size

if __name__ == '__main__':
    dataset = TumorCellGraphDataset(
        dataset_root="/Users/zhangjiahao/Project/tic/data/example",
        node_features=['center_coord','SIZE','cell_type','biomarker_expression'],
        transform=mask_biomarker_expression
    )
    sampler = WeightedSubgraphSampler(dataset, k=3, batch_size=10, shuffle=True, cell_types = ['Tumor'])

    for batch in sampler:
        print(batch)