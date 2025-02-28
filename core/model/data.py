import logging
import os
import numpy as np
import pickle
import pandas as pd
import torch
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph
from core.model.feature import process_biomarker_expression
from spacegm.utils import BIOMARKERS_UPMC, CELL_TYPE_MAPPING_UPMC

class TumorCellGraphDataset(Dataset):
    """
    A dataset class for processing and managing tumor cell graph data.

    Args:
        dataset_root (str): The root directory containing the dataset.
        cell_types (list[str], optional): List of cell types to consider. Defaults to ['Tumor'].
        node_features (list[str]): List of node features to be considered.
        biomarker_list (list): List of all biomarker types in the dataset.
        cell_type_mapping (dict): Mapping of cell types for one-hot encoding.
        edge_features (list, optional): List of edge features to be considered. Defaults to None.
        transform (callable, optional): A function/transform to apply to the data.
        pre_transform (callable, optional): A function/transform to apply to the data before saving.
    """

    def __init__(self, dataset_root, cell_types = ['Tumor'], biomarker_list=None, cell_type_mapping=None
                , node_features=None, edge_features=None, transform=None, pre_transform=None):
        """
        Initializes the dataset class.
        """
        # Initialize dataset attributes
        self.dataset_root = dataset_root
        self.biomarker_list = biomarker_list or BIOMARKERS_UPMC
        self.cell_type_mapping = cell_type_mapping or CELL_TYPE_MAPPING_UPMC
        self.node_features = node_features or ['center_coord', 'SIZE', 'cell_type', 'biomarker_expression']
        self.edge_features = edge_features or ['distance', 'edge_type']
        self.transform = transform
        self.pre_transform = pre_transform

        # load region ids and create region-cell mapping
        self.region_ids = self._load_region_ids()
        self.region_cell_ids = {region_id: self.get_cell_ids(region_id, cell_types) for region_id in self.region_ids}

        self.feature_indices = self.describe_features()

        # Initialize subgraph cache directory
        self.subgraph_cache_dir = os.path.join(self.dataset_root, 'subgraph')
        os.makedirs(self.subgraph_cache_dir, exist_ok=True)


    def _load_region_ids(self):
        """
        Loads region ids from the 'graph' directory.

        Returns:
            list: List of region IDs in the dataset.
        """
        graph_dir = os.path.join(self.dataset_root, 'graph')
        try:
            region_ids = [path.split('.')[0] for path in os.listdir(graph_dir)]
        except FileNotFoundError:
            raise FileNotFoundError(f"Graph directory not found: {graph_dir}")
        return region_ids
    
    def get_cell_ids(self, region_id, cell_types=None):
        """
        Reads the cell coordinates file for the given region and returns a list of cell IDs.
        Optionally, filters the cell IDs by the specified cell types.

        Args:
            region_id (str): The region ID.
            cell_types (list, optional): A list of cell types to filter by (default is ['Tumor']).

        Returns:
            list: A list of cell IDs for the specified region, filtered by cell types if provided.
        """
        cell_data_file = os.path.join(self.dataset_root, 'voronoi', f'{region_id}.cell_types.csv')

        try:
            if cell_types is not None:
                cell_types_df = pd.read_csv(cell_data_file)
                filtered_cell_ids = cell_types_df[cell_types_df['CELL_TYPE'].isin(cell_types)]['CELL_ID'].tolist()
            else:
                # do not filter cell ids by cell types
                filtered_cell_ids = cell_types_df['CELL_ID'].tolist()
            return filtered_cell_ids
        except FileNotFoundError:
            raise FileNotFoundError(f"Cell type file not found for region {region_id}: {cell_data_file}")
        except KeyError:
            raise KeyError(f"Missing 'CELL_TYPE' column in {cell_data_file}")

    def process_graph(self, region_id):
        """
        Process a single region's data to create a graph and include cell_id as a feature.
        
        Args:
            region_id (str): The region ID for which to create the graph.
        
        Returns:
            Data: PyG Data object containing the region's graph information.
        """
        graph_file = self._get_graph_file_path(region_id)
        G = self._load_graph_from_file(graph_file)

        # Extract features
        node_features = self.extract_node_features(G)
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
        edge_attr = self.extract_edge_features(G)

        # Add additional node features
        data = self._create_data_object(G, node_features, edge_index, edge_attr)

        # Apply pre-transform if exists
        if self.pre_transform:
            data = self.pre_transform(data)

        return data

    def _get_graph_file_path(self, region_id):
        """
        Returns the path to the graph file for a given region.
        """
        return os.path.join(self.dataset_root, 'graph', f'{region_id}.gpkl')

    def _load_graph_from_file(self, graph_file):
        """
        Loads the graph data from the specified file.

        Args:
            graph_file (str): The path to the graph file.

        Returns:
            networkx.Graph: The graph object loaded from the file.
        """
        try:
            with open(graph_file, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Graph file not found: {graph_file}")
        except pickle.UnpicklingError:
            raise ValueError(f"Error unpickling graph file: {graph_file}")

    def _create_data_object(self, G, node_features, edge_index, edge_attr):
        """
        Creates a PyTorch Geometric Data object containing the graph's node and edge features.

        Args:
            G (networkx.Graph): The graph object.
            node_features (torch.Tensor): The node features.
            edge_index (torch.Tensor): The edge index.
            edge_attr (torch.Tensor): The edge attributes.

        Returns:
            Data: The PyG Data object.
        """
        # Additional node features like cell_id and position
        cell_ids = torch.tensor([data['cell_id'] for _, data in G.nodes(data=True)], dtype=torch.long)
        cell_id_to_node_idx = {data['cell_id']: idx for idx, (node, data) in enumerate(G.nodes(data=True))}
        pos = torch.tensor([data['center_coord'] for _, data in G.nodes(data=True)], dtype=torch.float32)

        # Create PyG Data object
        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, cell_id=cell_ids, pos=pos)
        data.cell_id_to_node_idx = cell_id_to_node_idx
        data.feature_indices = self.feature_indices

        return data

    def describe_features(self):
        """
        Describes the feature indices for each feature type in the dataset.

        Returns:
            dict: Dictionary containing feature names and their corresponding indices.
        """
        feature_indices = {}
        total_dim = 0

        feature_indices = self._describe_node_features(feature_indices, total_dim)
        feature_indices['total_dim'] = total_dim
        return feature_indices

    def _describe_node_features(self, feature_indices, total_dim):
        """
        Describes and updates feature indices for node features.

        Args:
            feature_indices (dict): Dictionary to store feature indices.
            total_dim (int): The total dimension for all features.

        Returns:
            dict: Updated feature indices.
        """
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

        return feature_indices

    def extract_node_features(self, G):
        """
        Extract node features based on the provided node feature list.
        
        Args:
            G (networkx.Graph): The graph object containing the node features.

        Returns:
            torch.Tensor: Node features as a tensor.
        """
        node_features = []
        for node, data in G.nodes(data=True):
            feature_list = []

            feature_list = self._extract_feature_data(data, feature_list)

            node_features.append(feature_list)

        return torch.tensor(node_features, dtype=torch.float32)

    def _extract_feature_data(self, data, feature_list):
        """
        Extracts feature data for a node.

        Args:
            data (dict): The node data containing features.
            feature_list (list): List to store extracted features.

        Returns:
            list: Updated feature list.
        """
        if 'center_coord' in self.node_features:
            feature_list.extend(np.array(data['center_coord']).flatten())

        if 'biomarker_expression' in self.node_features:
            biomarker_expr = np.array([data['biomarker_expression'].get(bm, 0) for bm in self.biomarker_list])
            biomarker_expr = process_biomarker_expression(biomarker_expr)  # Apply predefined processing function
            feature_list.extend(biomarker_expr)

        if 'SIZE' in self.node_features:
            feature_list.append(data['SIZE'])

        if 'cell_type' in self.node_features:
            cell_type_index = self.cell_type_mapping.get(data['cell_type'], -1)
            cell_type_one_hot = np.zeros(len(self.cell_type_mapping))
            if cell_type_index != -1:
                cell_type_one_hot[cell_type_index] = 1
            feature_list.extend(cell_type_one_hot)

        return feature_list

    def extract_edge_features(self, G):
        """
        Extract edge features based on edge attributes in the graph.
        
        Args:
            G (networkx.Graph): The graph object containing the edge features.

        Returns:
            torch.Tensor: Edge features as a tensor.
        """
        edge_attr = []
        for _, _, data in G.edges(data=True):
            edge_feature = []

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

        if os.path.exists(cache_filename):
            with open(cache_filename, 'rb') as f:
                
                subgraph_data = pickle.load(f)
                if self.transform:
                    subgraph_data = self.transform(subgraph_data)
                return subgraph_data

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
            feature_indices=self.feature_indices,
            region_id=region_id,
            cell_id=cell_id,
            subset=subset, # node indices in the subgraph, relative to the original graph. 
        )

        with open(cache_filename, 'wb') as f:
            pickle.dump(subgraph_data, f)

        # Apply transform if exists
        if self.transform:
            subgraph_data = self.transform(subgraph_data)
            
        return subgraph_data

    def __getitem__(self, idx):
        """
        Retrieves a graph from the dataset using the index.
        """
        region_id = self.region_ids[idx]
        return self.process_graph(region_id)

    def __len__(self):
        """
        Returns the total number of regions in the dataset.
        """
        return len(self.region_ids)

    def clear_cache(self):
        """Clears the subgraph cache directory."""
        for filename in os.listdir(self.subgraph_cache_dir):
            file_path = os.path.join(self.subgraph_cache_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

class RegionCellSubgraphDataset(Dataset):
    """
    A custom dataset for fetching subgraphs for each region-cell pair.
    """
    def __init__(self, dataset, region_cell_ids, log_file="subgraph_errors.log"):
        """
        Args:
            dataset (TumorCellGraphDataset): The TumorCellGraphDataset instance containing the data.
            region_cell_ids (dict): Dictionary of region IDs and corresponding cell IDs.
            log_file (str, optional): The file where errors will be logged. Default is 'subgraph_errors.log'.
        """
        self.dataset = dataset
        self.region_cell_ids = region_cell_ids

        # Generate all region-cell pairs (region_id, cell_id)
        self.region_cell_pairs = [(region_id, cell_id) for region_id, cell_ids in region_cell_ids.items() for cell_id in cell_ids]

        # Set up the logger with the specified log file
        logging.basicConfig(filename=log_file, level=logging.ERROR)
        self.logger = logging.getLogger()

    def __getitem__(self, idx):
        """
        Retrieves the subgraph for the specified region-cell pair.
        """
        region_id, cell_id = self.region_cell_pairs[idx]
        
        if region_id not in self.dataset.region_cell_ids:
            # Log missing region and skip
            self.logger.error(f"Region ID {region_id} not found in dataset.")
            return self.get_empty_graph()  # Return a valid empty graph
        
        if cell_id not in self.dataset.region_cell_ids[region_id]:
            # Log missing cell in region and skip
            self.logger.error(f"Cell ID {cell_id} not found for Region ID {region_id}.")
            return self.get_empty_graph()  # Return a valid empty graph

        try:
            # Try to fetch the subgraph for the given region and cell
            subgraph_data = self.dataset.get_subgraph(region_id, cell_id)
            
            if subgraph_data is None:
                # If no subgraph is found, log the error and return a valid empty graph
                self.logger.error(f"Subgraph for Region ID {region_id}, Cell ID {cell_id} is None.")
                return self.get_empty_graph()  # Return a valid empty graph
                
            return subgraph_data
        
        except KeyError as e:
            # Log the error and skip this pair if the region/cell is not found
            self.logger.error(f"KeyError: Region ID {region_id}, Cell ID {cell_id} not found. Error: {str(e)}")
            return self.get_empty_graph()  # Return a valid empty graph

    def __len__(self):
        """
        Returns the total number of region-cell pairs.
        """
        return len(self.region_cell_pairs)

    def get_empty_graph(self):
        """
        Returns a valid empty graph.
        """
        return Data(x=torch.empty(0, 0), edge_index=torch.empty(2, 0), y=torch.empty(0))

def get_region_cell_subgraph_dataloader(dataset, batch_size=1, shuffle=True,log_file="subgraph_errors.log"):
    """
    Returns a DataLoader that iterates through all region-cell pair subgraphs.
    
    Args:
        dataset (TumorCellGraphDataset): The dataset containing the data.
        batch_size (int, optional): The batch size for the DataLoader. Default is 1.
        shuffle (bool, optional): Whether to shuffle the dataset. Default is True.

    Returns:
        DataLoader: A DataLoader for iterating over region-cell subgraphs.
    """

    region_cell_subgraph_dataset = RegionCellSubgraphDataset(dataset, dataset.region_cell_ids,log_file)

    def custom_collate_fn(batch):
        # Filter out invalid or empty graphs
        batch = [data for data in batch if data is not None and len(data.x) > 0 and len(data.edge_index) > 0]
        if len(batch) == 0:
            return Batch()  
        return Batch.from_data_list(batch)

    return DataLoader(region_cell_subgraph_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn)