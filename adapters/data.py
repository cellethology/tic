import os
from spacegm.utils import BIOMARKERS_UPMC, CELL_TYPE_MAPPING_UPMC
import torch
import pickle
from torch_geometric.data import Dataset
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

import numpy as np
import torch

class TumorCellGraphDataset(Dataset):
    def __init__(self, dataset_root, node_features:list[str] = ['center_coord','SIZE','cell_type','biomarker_expression'], cell_types_list=BIOMARKERS_UPMC, cell_type_mapping=CELL_TYPE_MAPPING_UPMC, edge_features=None, transform=None, pre_transform=None):
        """
        Args:
            dataset_root (str): The root directory containing the dataset.
            cell_types_list (list): List of all cell types in the dataset (e.g., BIOMARKERS_UPMC).
            cell_type_mapping (dict): Mapping of cell types for one-hot encoding.
            node_features (list): List of node features to be considered.
            edge_features (list, optional): List of edge features to be considered. Defaults to None.
            transform (callable, optional): A function/transform to apply to the data.
            pre_transform (callable, optional): A function/transform to apply to the data before saving.
        """
        self.dataset_root = dataset_root
        self.cell_types_list = cell_types_list
        self.cell_type_mapping = cell_type_mapping
        self.node_features = node_features
        self.edge_features = edge_features if edge_features is not None else ['distance', 'edge_type']  # Default to all features from G
        self.transform = transform
        self.pre_transform = pre_transform
        self.region_ids = [path.split('.')[0] for path in os.listdir(os.path.join(dataset_root, 'graph'))]

        self.describe()

    def process_graph(self, region_id):
        """
        Process a single region's data to create a graph and include cell_id as a feature.
        
        Args:
            region_id (str): The region ID for which to create the graph.
        
        Returns:
            Data: PyG Data object containing the region's graph information.
        """
        # Load preprocessed graph from 'graph/' directory
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

        # Apply transformations if any
        if self.transform:
            data = self.transform(data)
        if self.pre_transform:
            data = self.pre_transform(data)

        return data
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
                biomarker_expr = np.array([data['biomarker_expression'].get(bm, 0) for bm in self.node_features])
                feature_list.extend(biomarker_expr)

            # Add SIZE if requested
            if 'SIZE' in self.node_features:
                feature_list.append(data['SIZE'])  # This is a scalar, so it's naturally a 1D array of length 1
            
            # Add cell type as one-hot encoding if requested
            if 'cell_type' in self.node_features:
                cell_type_index = self.cell_type_mapping.get(data['cell_type'], -1)
                cell_type_one_hot = np.zeros(len(self.cell_types_list))
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
            torch.Tensor: Edge features as a tensor.
        """
        edge_attr = []
        for _, _, data in G.edges(data=True):
            edge_feature = []
            # Add edge-specific features based on the list of required edge features
            for feature in self.edge_features:
                if feature == 'distance':
                    edge_feature.append(data['distance'])
                elif feature == 'edge_type':
                    edge_feature.append(1 if data['edge_type'] == 'neighbor' else 0)  # One-hot encoding for edge_type
            edge_attr.append(edge_feature)
        
        return torch.tensor(edge_attr, dtype=torch.float32)

    def get_subgraph(self, region_id, cell_id, k=3):
        """
        Get a k-hop subgraph for a specific cell_id in a region.
        Notion:
            according to pyG k-hop subgraph, the mapping index is always n-1 (n is the number of nodes in the subgraph)
        """
        data = self.process_graph(region_id)

        # Get the node index corresponding to the given cell_id
        node_idx = data.cell_id_to_node_idx[cell_id]

        # Get the k-hop subgraph using PyG's k_hop_subgraph utility
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx, k, data.edge_index, relabel_nodes=False)

        # Get node features for the subgraph
        node_features = data.x[edge_index[0]]

        # Get edge features for the subgraph
        edge_attr = data.edge_attr[edge_index[0]]

        # Get the positions (pos) for the subgraph
        pos = data.pos[edge_index[0]]

        # Create a new PyG Data object for the subgraph
        subgraph_data = Data(
            x=node_features, 
            edge_index=edge_index, 
            edge_attr=edge_attr,
            y=None,  
            pos=pos,  
            # Add additional information to the subgraph data object
            global_node_idx=torch.Tensor(node_idx),  # Store the global node index for the center node
            relabeled_node_idx=mapping  # Store the relabeled node index
        )

        return subgraph_data

    def describe(self):
        """
        Describe the structure of the node feature tensor (x):
        1. Total dimension of x.
        2. Dimension ranges for each feature (e.g., center_coord occupies dimension 0-1, biomarker_expression occupies 2-<len>).
        """
        feature_info = {}

        # Calculate the total dimension
        total_dim = 0
        feature_info['total_dim'] = total_dim

        # Add dimensions based on requested node features
        if 'center_coord' in self.node_features:
            feature_info['center_coord'] = (total_dim, total_dim + 1)
            total_dim += 2  # center_coord is a 2D feature (X, Y)
        
        if 'biomarker_expression' in self.node_features:
            biomarker_start = total_dim
            biomarker_end = total_dim + len(self.node_features)
            feature_info['biomarker_expression'] = (biomarker_start, biomarker_end)
            total_dim += len(self.node_features)  # Number of biomarkers selected

        if 'SIZE' in self.node_features:
            feature_info['SIZE'] = (total_dim, total_dim + 1)
            total_dim += 1  # SIZE is a single feature
        
        if 'cell_type' in self.node_features:
            cell_type_start = total_dim
            cell_type_end = total_dim + len(self.cell_types_list)
            feature_info['cell_type'] = (cell_type_start, cell_type_end)
            total_dim += len(self.cell_types_list)  # One-hot encoding for cell type
        
        # Final total dimension
        feature_info['total_dim'] = total_dim
        print(feature_info)
        return feature_info

    def __getitem__(self, idx):
        region_id = self.region_ids[idx]
        return self.process_graph(region_id)

    def __len__(self):
        return len(self.region_ids)

if __name__ == '__main__':
    dataset = TumorCellGraphDataset(dataset_root = "/Users/zhangjiahao/Project/tic/data/example",
                                    node_features = ['center_coord','biomarker_expression','cell_type']
                                    )
    example_data = dataset[0]   
    subgraph = dataset.get_subgraph('UPMC_c001_v001_r001_reg001', 99, k=3)
    print(subgraph)