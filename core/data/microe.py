import numpy as np
import torch
from torch_geometric.data import Data

from core.constant import ALL_BIOMARKERS, ALL_CELL_TYPES

class MicroE:
    """
    Represents the microenvironment centered on a specific cell.
    
    This class encapsulates the local environment around a center cell,
    including its neighboring cells and an optional PyG graph representation.
    
    Key functionalities:
      - Store the center cell and its neighbor cells.
      - Provide access interfaces for the center cell and neighbors.
      - Convert the microenvironment into a PyG graph using provided feature functions.
    """
    def __init__(self, center_cell, neighbors, graph=None):
        """
        Initialize a MicroE object.
        
        :param center_cell: The central Cell object.
        :param neighbors: A list of Cell objects representing neighboring cells.
        :param graph: Optional precomputed PyG graph (Data object) for the microenvironment.
        """
        self.center_cell = center_cell
        self.neighbors = neighbors  # Neighboring cells list
        self.cells = [center_cell] + neighbors  # Complete list (center + neighbors)
        self.graph = graph  # Precomputed PyG graph (if available)
    
    def get_center_cell(self):
        """
        Return the center cell of the microenvironment.
        
        :return: The center Cell object.
        """
        return self.center_cell
    
    def get_neighbors(self):
        """
        Return the list of neighboring cells in the microenvironment.
        
        :return: A list of Cell objects.
        """
        return self.neighbors
    
    def to_graph(self, node_feature_fn, edge_index_fn, edge_attr_fn=None):
        """
        Convert the microenvironment into a graph structure compatible with PyTorch Geometric (PyG).
        
        If a precomputed graph exists, it will be returned directly. Otherwise,
        the graph is generated using the provided feature functions.
        
        :param node_feature_fn: A function that takes a Cell object and returns its node features.
        :param edge_index_fn: A function that takes a list of Cell objects and returns an edge index tensor.
        :param edge_attr_fn: Optional function that takes two Cell objects and returns edge attributes.
        :return: A PyG Data object representing the microenvironment graph.
        """
        if self.graph is not None:
            return self.graph
        
        # Generate node features for all cells in the microenvironment.
        node_features = torch.tensor([node_feature_fn(cell) for cell in self.cells], dtype=torch.float)
        
        # Generate edge index based on the cell list.
        edge_index = edge_index_fn(self.cells)
        
        # Optionally, generate edge attributes if a function is provided.
        edge_attr = None
        if edge_attr_fn is not None:
            # Iterate over each edge (i, j) in the edge index.
            edges = edge_index.t().tolist()
            edge_attr = torch.tensor([edge_attr_fn(self.cells[i], self.cells[j]) for i, j in edges], dtype=torch.float)
        
        self.graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        return self.graph
    
    def get_neighborhood_biomarker_matrix(self, biomarkers: list = ALL_BIOMARKERS, cell_types: list = ALL_CELL_TYPES) -> np.ndarray:
        """
        Generate a biomarker expression matrix for the neighborhood (excluding the center cell).
        
        The resulting matrix is of shape (N, M), where:
          - N is the total number of cell types (default: ALL_CELL_TYPES),
          - M is the total number of biomarkers (default: ALL_BIOMARKERS).
        
        For each cell type, if multiple neighbor cells exist, their biomarker expression values 
        are averaged across cells. If no cell of a specific type is present among the neighbors, 
        that row is filled with np.nan.
        
        :param biomarkers: Optional list of biomarker names. If None, ALL_BIOMARKERS is used.
        :param cell_types: Optional list of cell types. If None, ALL_CELL_TYPES is used.
        :return: A NumPy array of shape (N, M) representing the averaged biomarker expressions.
        """
        
        # Only consider neighbors (exclude the center cell).
        neighbor_cells = self.neighbors
        
        # Initialize the matrix with np.nan values.
        biomarker_matrix = np.full((len(cell_types), len(biomarkers)), np.nan, dtype=float)
        
        # For each cell type, compute the average biomarker values among neighbor cells.
        for i, ctype in enumerate(cell_types):
            # Select cells with matching type.
            cells_of_type = [cell for cell in neighbor_cells if cell.cell_type == ctype]
            if cells_of_type:
                for j, biomarker in enumerate(biomarkers):
                    # Gather biomarker values; skip missing values.
                    values = [cell.get_biomarker(biomarker) for cell in cells_of_type if cell.get_biomarker(biomarker) is not None]
                    if values:
                        biomarker_matrix[i, j] = np.mean(values)
        
        return biomarker_matrix

    def __str__(self):
        return f"Microenvironment around Cell {self.center_cell.cell_id} with {len(self.neighbors)} neighbors"