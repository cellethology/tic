import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from core.constant import ALL_BIOMARKERS, ALL_CELL_TYPES, DEFAULT_REPRESENTATION_PIPELINE, REPRESENTATION_METHODS
from core.data.cell import Cell

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
    def __init__(self, center_cell, neighbors, tissue_id, graph=None):
        """
        Initialize a MicroE object.
        
        :param center_cell: The central Cell object.
        :param neighbors: A list of Cell objects representing neighboring cells.
        :param graph: Optional precomputed PyG graph (Data object) for the microenvironment.
        """
        self.center_cell = center_cell
        self.neighbors = neighbors  # Neighboring cells list
        self.cells = [center_cell] + neighbors  # Complete list (center + neighbors)
        self.tissue_id = tissue_id
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
    
    # ---------------------------------------------------------------------
    # Internal functions for each representation method
    # ---------------------------------------------------------------------
    def _get_raw_expression(self, biomarkers=ALL_BIOMARKERS):
        """Directly use the center cell's biomarker expression as a vector."""
        return np.array([
            self.center_cell.get_biomarker(bm) for bm in biomarkers
        ], dtype=float)

    def _get_neighbor_composition(self, cell_types=ALL_CELL_TYPES):
        """Use neighbor composition (fraction of each cell type) as a vector."""
        total_neighbors = len(self.neighbors)
        counts = []
        for ctype in cell_types:
            n = sum(1 for c in self.neighbors if c.cell_type == ctype)
            counts.append(n / total_neighbors if total_neighbors > 0 else 0.0)
        return np.array(counts, dtype=float)

    def _get_nn_embedding(self, model: nn.Module, device: torch.device):
        """Use a neural network to get an embedding from the PyG graph."""
        if self.graph is None:
            raise ValueError("No PyG graph found. Build or assign self.graph first.")

        model.eval()
        graph_on_device = self.graph.to(device)
        with torch.no_grad():
            embedding = model(graph_on_device)
        return embedding.cpu().numpy()

    # A dictionary mapping method name -> the internal function
    _REPRESENTATION_FUNCS = {
        REPRESENTATION_METHODS["raw_expression"]: _get_raw_expression,
        REPRESENTATION_METHODS["neighbor_composition"]: _get_neighbor_composition,
        REPRESENTATION_METHODS["nn_embedding"]: _get_nn_embedding
    }

    # ---------------------------------------------------------------------
    # Public method to export the center cell with chosen representations
    # ---------------------------------------------------------------------
    def export_center_cell_with_representations(
        self,
        representations=DEFAULT_REPRESENTATION_PIPELINE,
        model: nn.Module = None,
        device: torch.device = None,
        biomarkers: list = ALL_BIOMARKERS,
        cell_types: list = ALL_CELL_TYPES
    ) -> Cell:
        """
        Attach the selected representation vectors to the center cell
        and return the center cell object. By default, uses the
        DEFAULT_REPRESENTATION_PIPELINE from constant.py.
        
        :param representations: A list of method names (e.g. ['raw_expression', 'neighbor_composition']).
                               If None, uses DEFAULT_REPRESENTATION_PIPELINE.
        :param model: Optional PyTorch NN model for 'nn_embedding'.
        :param device: The device to run the model on (if using 'nn_embedding').
        :param biomarkers: The biomarkers to consider for 'raw_expression'.
        :param cell_types: The cell types to consider for 'neighbor_composition'.
        :return: The center cell object with new features in additional_features.
        """
        if representations is None:
            representations = DEFAULT_REPRESENTATION_PIPELINE
        
        for method_name in representations:
            func = self._REPRESENTATION_FUNCS.get(method_name, None)
            if func is None:
                print(f"Warning: Unknown representation method '{method_name}' - skipping.")
                continue
            
            # Depending on the method, we pass different arguments
            if method_name == REPRESENTATION_METHODS["raw_expression"]:
                rep_vec = func(self, biomarkers=biomarkers)
            elif method_name == REPRESENTATION_METHODS["neighbor_composition"]:
                rep_vec = func(self, cell_types=cell_types)
            elif method_name == REPRESENTATION_METHODS["nn_embedding"]:
                if model is None or device is None:
                    print("Warning: 'nn_embedding' requires model and device. Skipping.")
                    continue
                rep_vec = func(self, model=model, device=device)
            else:
                print(f"Warning: method '{method_name}' not implemented.")
                continue
            
            # Attach the representation to the center cell
            self.center_cell.add_feature(method_name, rep_vec)
        
        return self.center_cell

    # ---------------------------------------------------------------------
    # Public methods for causal inference
    # ---------------------------------------------------------------------
    def get_center_biomarker_vector(self, y_biomarkers):
        """
        Retrieve one or multiple biomarkers from the center cell.
        
        :param y_biomarkers: A single biomarker name (str) or list of biomarker names.
        :return: A 1D NumPy array of shape (len(y_biomarkers),).
        """
        if isinstance(y_biomarkers, str):
            y_biomarkers = [y_biomarkers]
        return np.array(
            [self.center_cell.get_biomarker(bm) for bm in y_biomarkers],
            dtype=float
        )

    def prepare_for_causal_inference(
        self,
        y_biomarkers,
        x_biomarkers=ALL_BIOMARKERS,
        x_cell_types=ALL_CELL_TYPES
    ):
        """
        Prepare the data needed for causal inference. By default, X uses all biomarkers
        and all cell types in the microenvironment. Y can be one or more chosen biomarkers
        from the center cell.
        
        :param y_biomarkers: A single biomarker name (str) or list of biomarker names for the outcome.
        :param x_biomarkers: List of biomarker names to be used for X.
        :param x_cell_types: List of cell types to be used for X.
        :return: (X, Y, X_labels, Y_labels)
                 - X: 1D NumPy array of shape (# cell_types * # biomarkers,)
                 - Y: 1D NumPy array of shape (len(y_biomarkers),)
                 - X_labels: A list of strings naming the X columns as f"{bm}&{ct}"
                 - Y_labels: A list of strings for the selected Y biomarker names
        """
        # 1) Get neighbor biomarker matrix
        neighborhood_matrix = self.get_neighborhood_biomarker_matrix(
            biomarkers=x_biomarkers, 
            cell_types=x_cell_types
        )
        # Flatten to 1D
        X = neighborhood_matrix.flatten()
        
        # 2) Create labels for X in the format "biomarker&celltype"
        X_labels = []
        for ct in x_cell_types:
            for bm in x_biomarkers:
                X_labels.append(f"{bm}&{ct}")
        
        # 3) Get center cell biomarkers for Y
        Y = self.get_center_biomarker_vector(y_biomarkers)
        
        # If y_biomarkers is a single str, we already converted it into a list in get_center_biomarker_vector
        if isinstance(y_biomarkers, str):
            Y_labels = [y_biomarkers]
        else:
            Y_labels = y_biomarkers
        
        return X, Y, X_labels, Y_labels

    def __str__(self):
        return f"Microenvironment around Cell {self.center_cell.cell_id} with {len(self.neighbors)} neighbors"