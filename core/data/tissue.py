import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph

from core.constant import MICROE_NEIGHBOR_CUTOFF
from core.data.microe import MicroE

class Tissue:
    """
    A class to represent a tissue sample, containing a list of cells, and optionally a precomputed PyG graph.
    
    The Tissue class stores all the cells within a region, provides methods to extract cell information,
    perform statistical analysis, and convert the tissue into a graph structure for use with PyTorch Geometric.
    
    This version supports:
      - Storing and reusing a precomputed PyG graph (to avoid expensive recomputation).
      - Saving the graph to disk and loading it later to instantiate a Tissue instance.
    """
    def __init__(self, tissue_id, cells, position=None, graph=None):
        """
        Initialize a Tissue object.
        
        :param tissue_id: The unique identifier for the tissue sample.
        :param cells: A list of Cell objects that make up the tissue.
        :param position: The spatial position of the tissue. Defaults to (0, 0).
        :param graph: Optional precomputed PyG graph (Data object). Defaults to None.
        """
        self.tissue_id = tissue_id
        self.cells = cells  # List of Cell objects
        self.cell_dict = {cell.cell_id: cell for cell in cells}  # O(1) lookup for cells by ID
        self.pos = position if position else (0, 0)

        # Precompute positions array for all cells (for vectorized computations)
        self.positions = np.array([cell.pos for cell in self.cells])
        self.validate_cells_positions()

        # Placeholder for the PyG graph. If provided, reuse it.
        self.graph = graph

    def validate_cells_positions(self):
        """
        Ensures that all cells in the tissue have the same number of dimensions in their position data.
        """
        if self.positions.ndim != 2:
            raise ValueError("Cell positions should be a 2D array.")
        
        expected_dim = self.positions.shape[1]
        if not all(len(cell.pos) == expected_dim for cell in self.cells):
            raise ValueError("Inconsistent position dimensions in cell data.")
        
    def get_cell_by_id(self, cell_id):
        """
        Retrieve a cell by its unique ID.
        
        :param cell_id: The unique identifier for the cell.
        :return: The Cell object, or None if not found.
        """
        return self.cell_dict.get(cell_id, None)
    
    def get_microenvironment(self, center_cell_id: str, k: int = 3, microe_neighbor_cutoff: float = MICROE_NEIGHBOR_CUTOFF) -> MicroE:
        """
        Extract the k-hop microenvironment for the specified center cell using the precomputed Tissue graph.
        
        This function leverages the PyG graph (whose edge_index is based on node indices) to extract a subgraph 
        corresponding to the microenvironment, including node features, edge indices, and edge attributes.
        Additionally, it filters the microenvironment so that only cells within a distance threshold 
        (MICROE_NEIGHBOR_CUTOFF) from the center cell are retained.
        
        :param center_cell_id: The unique identifier for the center cell.
        :param k: The number of hops (k-hop subgraph) to include in the microenvironment.
        :param microe_neighbor_cutoff: The distance threshold for filtering neighboring cells.
        :return: A MicroE object representing the microenvironment, with its graph attribute set.
        :raises ValueError: If the Tissue graph is not computed or the center cell is not found.
        """
        if self.graph is None:
            raise ValueError("Tissue graph has not been computed. Please call to_graph() first.")

        # Find the index of the center cell in self.cells (node ordering is 0, 1, 2, ..., N-1)
        center_index = None
        for idx, cell in enumerate(self.cells):
            if cell.cell_id == center_cell_id:
                center_index = idx
                break
        if center_index is None:
            raise ValueError("Center cell not found in Tissue.")

        # Directly extract the k-hop subgraph using the Tissue's edge_index.
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            center_index, num_hops=k, edge_index=self.graph.edge_index,
            relabel_nodes=True, num_nodes=len(self.cells)
        )

        # Extract edge attributes if available using the edge_mask.
        if hasattr(self.graph, 'edge_attr') and self.graph.edge_attr is not None:
            sub_edge_attr = self.graph.edge_attr[edge_mask]
        else:
            sub_edge_attr = None

        # Retrieve the list of cells corresponding to the nodes in the subgraph.
        micro_cells = [self.cells[i] for i in subset.tolist()]
        center_cell = self.get_cell_by_id(center_cell_id)

        # Instantiate a MicroE object using the extracted cells.
        micro_env = MicroE(center_cell, micro_cells, graph=None)
        micro_env.graph = Data(x=self.graph.x[subset], edge_index=sub_edge_index, edge_attr=sub_edge_attr)

        # --- Filtering: Retain only cells within MICROE_NEIGHBOR_CUTOFF from the center cell ---
        # For each cell in the microenvironment, compute its Euclidean distance from the center cell.
        filtered_indices = []
        for i, cell in enumerate(micro_cells):
            dist = np.linalg.norm(np.array(cell.pos) - np.array(center_cell.pos))
            if dist <= microe_neighbor_cutoff:
                filtered_indices.append(i)

        # Ensure the center cell is included.
        if not any(cell.cell_id == center_cell.cell_id for i, cell in enumerate(micro_cells) if i in filtered_indices):
            for i, cell in enumerate(micro_cells):
                if cell.cell_id == center_cell.cell_id:
                    filtered_indices.append(i)
                    break

        # Convert the filtered indices to a tensor.
        filtered_indices_tensor = torch.tensor(filtered_indices, dtype=torch.long)
        # Induce a subgraph from the current microenvironment graph using the filtered node indices.
        new_edge_index, new_edge_attr = subgraph(
            filtered_indices_tensor, micro_env.graph.edge_index, micro_env.graph.edge_attr,
            relabel_nodes=True, num_nodes=micro_env.graph.num_nodes
        )
        new_x = micro_env.graph.x[filtered_indices_tensor]

        # Update the MicroE graph with the filtered subgraph.
        micro_env.graph = Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr)
        
        # Update the cell lists accordingly.
        filtered_micro_cells = [micro_cells[i] for i in filtered_indices]
        # Neighbors: all filtered cells except the center cell.
        filtered_neighbors = [cell for cell in filtered_micro_cells if cell.cell_id != center_cell.cell_id]
        micro_env.neighbors = filtered_neighbors
        micro_env.cells = filtered_micro_cells

        return micro_env

    def get_biomarkers_of_all_cells(self, biomarker_name):
        """
        Retrieve the expression levels of a specified biomarker for all cells in the tissue.
        
        :param biomarker_name: The name of the biomarker (e.g., 'PanCK').
        :return: A dictionary with cell IDs as keys and the biomarker expression levels as values.
        """
        return {cell.cell_id: cell.get_biomarker(biomarker_name) for cell in self.cells}

    def get_statistics_for_biomarker(self, biomarker_name):
        """
        Calculate statistical information for a biomarker across all cells.
        
        :param biomarker_name: The name of the biomarker to analyze.
        :return: A tuple containing the mean and standard deviation of the biomarker expression levels.
        """
        biomarker_values = [cell.get_biomarker(biomarker_name) for cell in self.cells 
                            if cell.get_biomarker(biomarker_name) is not None]
        
        if not biomarker_values:
            raise ValueError(f"No data available for biomarker '{biomarker_name}'.")
        
        return np.mean(biomarker_values), np.std(biomarker_values)

    def to_graph(self, node_feature_fn: callable, edge_index_fn: callable, edge_attr_fn=None):
        """
        Convert the tissue into a graph structure compatible with PyTorch Geometric (PyG).
        
        If a precomputed graph is already stored in this Tissue instance, it will be returned directly.
        Otherwise, the graph is computed, stored, and returned.
        
        :param node_feature_fn: A function that takes a Cell object and returns a list of features for the node.
        :param edge_index_fn: A function that takes a list of Cell objects and returns an edge index tensor.
        :param edge_attr_fn: An optional function that takes two Cell objects and returns the edge attribute.
        :return: A PyG Data object representing the tissue as a graph.
        """
        # Return the precomputed graph if available
        if self.graph is not None:
            return self.graph

        # Generate node features from each cell
        node_features = self._generate_node_features(node_feature_fn)

        # Generate edge index using the provided function
        edge_index = edge_index_fn(self.cells)

        # Generate edge attributes if an edge attribute function is provided
        edge_attr = self._generate_edge_attributes(edge_attr_fn, edge_index)

        # Create a PyG Data object
        self.graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        return self.graph

    def _generate_node_features(self, node_feature_fn):
        """
        Generate the node features by applying the given node_feature_fn to each cell.
        Processes all cells in a single step using list comprehension.
        
        :param node_feature_fn: A function that generates node features from each cell.
        :return: A torch tensor of node features.
        """
        try:
            node_features = torch.tensor([node_feature_fn(cell) for cell in self.cells], dtype=torch.float)
            return node_features
        except Exception as e:
            raise ValueError(f"Error generating node features: {e}")

    def _generate_edge_attributes(self, edge_attr_fn, edge_index):
        """
        Generate edge attributes by applying the edge_attr_fn to each pair of cells specified by edge_index.
        
        :param edge_attr_fn: A function that generates an edge attribute for each pair of cells.
        :param edge_index: The edge index tensor specifying connections between nodes.
        :return: A torch tensor of edge attributes, or None if edge_attr_fn is None.
        """
        if edge_attr_fn is None:
            return None  # No edge attributes are needed
        
        edge_attr = []
        # Iterate over each edge (source, target) pair in the edge_index tensor
        for edge in edge_index.t().tolist():
            cell1_index, cell2_index = edge
            cell1 = self.cells[cell1_index]
            cell2 = self.cells[cell2_index]
            if cell1 and cell2:
                edge_attr.append(edge_attr_fn(cell1, cell2))
            else:
                raise ValueError(f"Error retrieving cells for edge: {edge}")
        
        return torch.tensor(edge_attr, dtype=torch.float)

    def save_graph(self, filepath: str):
        """
        Save the precomputed PyG graph to disk.
        
        :param filepath: The file path to save the graph (e.g., 'graph.pt').
        :raises ValueError: If no graph has been computed yet.
        """
        if self.graph is None:
            raise ValueError("No precomputed graph available to save. Call to_graph() first.")
        torch.save(self.graph, filepath)

    @classmethod
    def load_graph(cls, tissue_id: str, cells: list, filepath: str, position=None):
        """
        Load a precomputed PyG graph from disk and instantiate a Tissue object with it.
        
        :param tissue_id: The unique identifier for the tissue sample.
        :param cells: A list of Cell objects associated with this tissue.
        :param filepath: The file path from which to load the graph.
        :param position: Optional spatial position of the tissue.
        :return: An instance of Tissue with the graph loaded.
        """
        graph = torch.load(filepath)
        return cls.from_pyg_graph(tissue_id, cells, graph, position)

    @classmethod
    def from_pyg_graph(cls, tissue_id: str, cells: list, pyg_graph: Data, position=None):
        """
        Instantiate a Tissue object using a precomputed PyG graph.
        
        :param tissue_id: The unique identifier for the tissue sample.
        :param cells: A list of Cell objects associated with this tissue.
        :param pyg_graph: A precomputed PyG graph (Data object).
        :param position: Optional spatial position of the tissue.
        :return: An instance of Tissue with the precomputed graph.
        """
        return cls(tissue_id, cells, position, graph=pyg_graph)

    def __str__(self):
        """
        Return a string representation of the Tissue object, summarizing the tissue ID and number of cells.
        """
        return f"Tissue {self.tissue_id} with {len(self.cells)} cells"