import os
import random
import numpy as np
import pickle
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph
from adapters.feature import process_biomarker_expression
from adapters.transform import mask_biomarker_expression
from adapters.utils import get_cell_ids_by_type
from spacegm.utils import BIOMARKERS_UPMC, CELL_TYPE_MAPPING_UPMC

class TumorCellGraphDataset(Dataset):
    def __init__(self, dataset_root, node_features:list[str] = ['center_coord','SIZE','cell_type','biomarker_expression'], biomarker_list=BIOMARKERS_UPMC, cell_type_mapping=CELL_TYPE_MAPPING_UPMC, edge_features=None, transform=None, pre_transform=None):
        """
        Args:
            dataset_root (str): The root directory containing the dataset.
            biomarker_list (list): List of all biomarker types in the dataset (e.g., BIOMARKERS_UPMC).
            cell_type_mapping (dict): Mapping of cell types for one-hot encoding.
            node_features (list): List of node features to be considered.
            edge_features (list, optional): List of edge features to be considered. Defaults to None.
            transform (callable, optional): A function/transform to apply to the data.
            pre_transform (callable, optional): A function/transform to apply to the data before saving.
        """
        self.dataset_root = dataset_root
        self.biomarker_list = biomarker_list
        self.cell_type_mapping = cell_type_mapping
        self.node_features = node_features
        self.edge_features = edge_features if edge_features is not None else ['distance', 'edge_type']  # Default to all features from G
        self.transform = transform
        self.pre_transform = pre_transform
        self.region_ids = [path.split('.')[0] for path in os.listdir(os.path.join(dataset_root, 'graph'))]

        self.feature_indices = self.describe()

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

        # Extract node features and store them in a list
        node_features = self.extract_node_features(G)
        
        # Extract edge index (from G) and edge attributes
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        edge_attr = self.extract_edge_features(G)

        # Add 'cell_id' as a feature for each node
        cell_ids = torch.tensor([data['cell_id'] for _, data in G.nodes(data=True)], dtype=torch.long)

        # Create a mapping from cell_id to node_idx
        cell_id_to_node_idx = {data['cell_id']: idx for idx, (node, data) in enumerate(G.nodes(data=True))}

        # Add 'pos' as a feature for each node
        pos = torch.tensor([data['center_coord'] for _, data in G.nodes(data=True)], dtype=torch.float32) # shape (n_nodes, 2) where each node has (X, Y)

        # Create PyG Data object
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, cell_id=cell_ids, pos=pos)

        # Store the cell_id to node index mapping in the data object
        data.cell_id_to_node_idx = cell_id_to_node_idx
        data.feature_indices = self.feature_indices

        if self.pre_transform:
            data = self.pre_transform(data)

        return data

    def describe(self):
        """
        Create a dictionary that stores the index ranges for each feature in the node features.
        
        Args:
            node_features (torch.Tensor): Node features matrix (n_nodes x num_features).
        
        Returns:
            dict: Dictionary containing feature names and their corresponding indices.

        Note:
            The indices are inclusive on the start and exclusive on the end.
            eg. If a feature has indices (0, 2), it means the feature is present at indices 0 and 1.
        """
        feature_indices = {}
        total_dim = 0

        if 'center_coord' in self.node_features:
            feature_indices['center_coord'] = (total_dim, total_dim + 2)
            total_dim += 2  # center_coord is a 2D feature (X, Y)
        
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
        print("Feature Indices:", feature_indices)
        return feature_indices
    
    def extract_node_features(self, G):
        """
        Extract node features from the graph G based on the input `node_features` list.
        
        Args:
            G (nx.Graph): The graph object containing the node features.
        
        Returns:
            torch.Tensor: Node features as a tensor.
        """
        node_features = []
        for node, data in G.nodes(data=True):
            # Start with an empty list of node features
            feature_list = []

            # Add center coordinates if requested
            if 'center_coord' in self.node_features:
                # Ensure center_coord is a 1D array
                feature_list.extend(np.array(data['center_coord']).flatten())  # Flatten to ensure it's 1D (X, Y)
            
            # Add biomarker expressions if requested
            if 'biomarker_expression' in self.node_features:
                # Only include biomarker expression features
                biomarker_expr = np.array([data['biomarker_expression'].get(bm, 0) for bm in self.biomarker_list])
                biomarker_expr = process_biomarker_expression(biomarker_expr)
                feature_list.extend(biomarker_expr)

            # Add SIZE if requested
            if 'SIZE' in self.node_features:
                feature_list.append(data['SIZE'])  # This is a scalar, so it's naturally a 1D array of length 1
            
            # Add cell type as one-hot encoding if requested
            if 'cell_type' in self.node_features:
                cell_type_index = self.cell_type_mapping.get(data['cell_type'], -1)
                cell_type_one_hot = np.zeros(len(self.cell_type_mapping.keys()))
                if cell_type_index != -1:
                    cell_type_one_hot[cell_type_index] = 1
                feature_list.extend(cell_type_one_hot)
            
            # Ensure that feature_list is correctly concatenated into a single 1D array
            node_features.append(feature_list)
    
        return torch.tensor(node_features, dtype=torch.float32)

    def extract_edge_features(self, G):
        """
        Extract edge features from the graph G.
        
        Args:
            G (nx.Graph): The graph object containing the edge features.
        
        Returns:
            torch.Tensor: Edge features as a tensor with shape (num_edges, num_features).
        """
        edge_attr = []
        for _, _, data in G.edges(data=True):
            edge_feature = []

            # Add edge-specific features with 'edge_type' always at index 0
            if 'edge_type' in self.edge_features:
                # One-hot encoding for edge_type; assume 'neighbor' is one class, and others are another
                edge_feature.append(1 if data['edge_type'] == 'neighbor' else 0)  # 0 for non-neighbor edges
            if 'distance' in self.edge_features:
                # Ensure 'distance' feature exists and is converted to float
                edge_feature.append(float(data['distance']))  # Ensure it's a float

            # Ensure the edge_feature is correctly formatted as a 1D tensor for each edge
            edge_attr.append(edge_feature)
        
        # Convert the list of edge features into a 2D tensor
        return torch.tensor(edge_attr, dtype=torch.float32)

    def get_subgraph(self, region_id, cell_id, k=3):
        """
        Get a k-hop subgraph for a specific cell_id in a region.
        Notion:
            According to PyG k-hop subgraph, the mapping index is always n-1 (n is the number of nodes in the subgraph)
        """
        data = self.process_graph(region_id)

        # Get the node index corresponding to the given cell_id
        node_idx = data.cell_id_to_node_idx[cell_id]

        # Get the k-hop subgraph using PyG's k_hop_subgraph utility
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx, k, data.edge_index, relabel_nodes=True)

        # Filter node features (x), edge attributes, and other node-related features using the relabeled indices
        node_features = data.x[subset]  # subset corresponds to the new node indices in the subgraph

        # Filter edge features (edge_attr) using the filtered edge_index
        edge_attr = data.edge_attr[edge_mask]  # Filter the edge attributes based on the edge mask

        # Filter positions (pos) for the subgraph
        pos = data.pos[subset]  # Use the subset of node indices for positions

        # Create a new PyG Data object for the subgraph with the filtered features
        subgraph_data = Data(
            x=node_features, 
            edge_index=edge_index, 
            edge_attr=edge_attr,
            y=None,  # You can add ground truth labels if needed
            pos=pos,  
            # additional features
            global_node_idx=torch.Tensor([node_idx]),  # Store the global node index for the center node
            relabeled_node_idx=mapping,  # Store the relabeled node index for the subgraph
            feature_indices=self.feature_indices,
            mask=None,  # Placeholder for biomarker mask, will be updated later
            batch=None  # batch index
        )

        # Apply transformations if any
        if self.transform:
            subgraph_data = self.transform(subgraph_data)

        return subgraph_data

    def __getitem__(self, idx):
        region_id = self.region_ids[idx]
        return self.process_graph(region_id)

    def __len__(self):
        return len(self.region_ids)

class SubgraphSampler(torch.utils.data.IterableDataset):
    def __init__(self, dataset, k=3, batch_size=32, shuffle=True, drop_last=False, cell_types=None, infinite=True):
        """
        Args:
            dataset (TumorCellGraphDataset): The dataset object containing graph data.
            k (int): The number of hops for the k-hop subgraph.
            batch_size (int): The number of subgraphs in each batch.
            shuffle (bool): Whether to shuffle the data before sampling.
            drop_last (bool): Whether to drop the last incomplete batch (if the data size is not divisible by batch size).
            cell_types (list, optional): A list of cell types to sample from. If None, all cell types are used.
            infinite (bool): Whether to keep iterating endlessly over the dataset.
        """
        self.dataset = dataset
        self.k = k
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.region_ids = self.dataset.region_ids
        self.cell_types = cell_types  # List of cell types to sample from
        self.infinite = infinite  # Flag for infinite iteration

        if self.shuffle:
            # Shuffle the indices of region_ids
            shuffled_indices = torch.randperm(len(self.region_ids)).tolist()
            self.region_ids = [self.region_ids[i] for i in shuffled_indices]  # Reorder region_ids based on shuffled indices

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
                # If we reach the end and drop_last is False, wrap around to the start
                if not self.drop_last:
                    remaining_ids = self.region_ids[idx:]
                    self.region_ids = self.region_ids[:idx]
                    self.region_ids.extend(remaining_ids)  # Wrap around and continue sampling
                    idx = 0
                else:
                    break  # Exit if drop_last is True and the final batch is incomplete

            batch_region_ids = self.region_ids[idx:idx + self.batch_size]
            batch_cell_ids = self.get_random_cell_ids(batch_region_ids)

            subgraphs = []
            for region_id, cell_id in zip(batch_region_ids, batch_cell_ids):
                subgraph = self.sample_subgraph(region_id, cell_id)
                subgraphs.append(subgraph)

            batch_data = Batch.from_data_list(subgraphs)
            yield batch_data

            idx += self.batch_size

    def get_random_cell_ids(self, batch_region_ids):
        """
        Randomly selects a cell_id for each region in the batch to serve as the center node for the subgraph.
        If cell_types are specified, only select cell_ids corresponding to those types.
        """
        batch_cell_ids = []
        for region_id in batch_region_ids:
            data = self.dataset.process_graph(region_id)  # Get the graph data for the region
            
            if self.cell_types:
                # Use the helper function to get the cell_ids based on the cell_types
                cell_ids = get_cell_ids_by_type(region_id, self.cell_types, self.dataset.dataset_root)
            else:
                # If no specific cell_types are given, select all cell_ids
                cell_ids = list(data.cell_id_to_node_idx.keys())
            
            # Randomly choose one cell_id as the center node
            batch_cell_ids.append(random.choice(cell_ids))

        return batch_cell_ids

    def sample_subgraph(self, region_id, cell_id):
        """
        Samples a k-hop subgraph for a given region and cell_id.
        """
        return self.dataset.get_subgraph(region_id, cell_id, self.k)
    
if __name__ == '__main__':
    dataset = TumorCellGraphDataset(
        dataset_root="/Users/zhangjiahao/Project/tic/data/example",
        node_features=['center_coord','SIZE','cell_type','biomarker_expression'],
        transform=mask_biomarker_expression
    )
    sampler = SubgraphSampler(dataset, k=3, batch_size=10, shuffle=True, cell_types = ['Tumor'])

    for batch in sampler:
        print(batch)