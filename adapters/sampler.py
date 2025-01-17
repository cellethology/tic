from abc import ABC, abstractmethod
from concurrent import futures
import pickle
import numpy as np
import os
import pandas as pd
import os
import random

from spacegm.data import CellularGraphDataset

class CustomSubgraphSampler:
    def __init__(self, raw_dir, seed=None):
        self.raw_dir = raw_dir
        self.cell_data = {}
        self.seed = seed  
        self.load_data()

    def load_data(self):
        """
        Load all necessary data from CSV files within the raw directory.
        """
        regions = [filename.split('.')[0] for filename in os.listdir(self.raw_dir) if 'cell_data.csv' in filename]
        for region_id in regions:
            cell_data_path = os.path.join(self.raw_dir, f'{region_id}.cell_data.csv')
            cell_types_path = os.path.join(self.raw_dir, f'{region_id}.cell_types.csv')
            expression_path = os.path.join(self.raw_dir, f'{region_id}.expression.csv')

            data = pd.read_csv(cell_data_path)
            types = pd.read_csv(cell_types_path)
            expression = pd.read_csv(expression_path)

            # Merge dataframes on 'CELL_ID'
            full_data = pd.merge(data, types, on='CELL_ID')
            full_data = pd.merge(full_data, expression, on='CELL_ID')
            
            self.cell_data[region_id] = full_data

    def sample(self, total_samples, regions='all', cell_types='all'):
        selected_cells = []
        region_list = self._select_regions(regions)
        for region_id in region_list:
            cells = self._filter_cells(region_id, cell_types)
            sampled_cells = self._sample_cells(cells, total_samples, len(region_list))
            selected_cells.extend([(region_id, cell['CELL_ID']) for _, cell in sampled_cells.iterrows()])
        return selected_cells

    def _select_regions(self, regions):
        if regions == 'all':
            return list(self.cell_data.keys())
        elif isinstance(regions, list):
            return regions
        else:
            raise ValueError("Invalid region specification.")

    def _filter_cells(self, region_id, cell_types):
        cells = self.cell_data[region_id]
        if cell_types == 'all':
            return cells
        else:
            return cells[cells['CLUSTER_LABEL'].isin(cell_types)]

    def _sample_cells(self, cells, total_samples, num_regions):
        if cells.empty:
            return pd.DataFrame()
        samples_per_region = total_samples // num_regions
        return cells.sample(min(len(cells), samples_per_region), random_state=self.seed)  
#------------------------------------------------------------
# Helper functions for retrieving subgraphs from the dataset
#------------------------------------------------------------

def get_subgraph_by_cell(dataset: CellularGraphDataset, region_id: str, cell_id: int):
    """
    Retrieve a subgraph centered at the specified cell_id within the given region_id.

    Args:
        region_id (str): The identifier of the region.
        cell_id (int): The identifier of the cell which will be the center of the subgraph.

    Returns:
        torch_geometric.data.Data: The subgraph centered at the specified cell.
    """
    # Find the index of the region in the dataset
    idx = dataset.region_ids.index(region_id)

    # Load the networkx graph from file
    graph_path = os.path.join(dataset.raw_dir, f"{dataset.region_ids[idx]}.gpkl")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Raw graph file not found for region: {dataset.region_ids[idx]}")

    nx_graph = pickle.load(open(graph_path, "rb"))

    # Locate the node corresponding to the given cell_id
    node_index = None
    for node, attrs in nx_graph.nodes(data=True):
        if attrs.get("cell_id") == cell_id:
            node_index = node
            break

    if node_index is None:
        raise ValueError(f"Cell ID {cell_id} not found in region {region_id}.")

    # Retrieve the PyG subgraph from the dataset using the node index found
    subgraph = dataset.get_subgraph(idx, node_index)
    return subgraph





