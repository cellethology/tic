import numpy as np
import torch
from torch_geometric.data import Data

class Tissue:
    """
    A class to represent a tissue sample, containing a list of cells.

    The Tissue class stores all the cells within a region, and provides methods to extract cell information, perform
    statistical analysis, and convert the tissue into a graph structure for use with PyTorch Geometric.
    """

    def __init__(self, tissue_id, cells, dimensions=None, position=None):
        """
        Initialize a Tissue object.

        :param tissue_id: The unique identifier for the tissue sample.
        :param cells: A list of Cell objects that make up the tissue.
        :param dimensions: The dimensions of the tissue (e.g., (2, 2) for 2D). Defaults to (2, 2).
        :param position: The spatial position of the tissue. Defaults to (0, 0).
        """
        self.tissue_id = tissue_id
        self.cells = cells  # List of Cell objects
        self.dimensions = dimensions if dimensions else (2, 2)  # Default to 2D
        self.position = position if position else (0, 0)  # Default position is (0, 0)

        # Validate that all cells have consistent position dimensions
        self.validate_cells_positions()

    def validate_cells_positions(self):
        """
        Ensures that all cells in the tissue have the same number of dimensions in their position data.
        """
        if not self.cells:
            raise ValueError("No cells in the tissue to validate.")

        # Get the number of dimensions from the first cell
        first_cell_pos_len = len(self.cells[0].pos)

        for cell in self.cells:
            if len(cell.pos) != first_cell_pos_len:
                raise ValueError(f"Cell position dimensions are inconsistent. Expected {first_cell_pos_len}-dimensional positions, found {len(cell.pos)} for cell {cell.cell_id}.")

    def get_cell_by_id(self, cell_id):
        """
        Retrieve a cell by its unique ID.

        :param cell_id: The unique identifier for the cell.
        :return: The Cell object, or None if not found.
        """
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell
        return None  # Return None if the cell is not found

    def get_biomarkers_of_all_cells(self, biomarker_name):
        """
        Retrieve the expression levels of a specified biomarker for all cells in the tissue.

        :param biomarker_name: The name of the biomarker (e.g., 'PanCK').
        :return: A dictionary with cell IDs as keys and the biomarker expression levels as values.
        """
        biomarker_data = {}
        for cell in self.cells:
            biomarker_data[cell.cell_id] = cell.get_biomarker(biomarker_name)
        return biomarker_data

    def get_statistics_for_biomarker(self, biomarker_name):
        """
        Calculate statistical information for a biomarker across all cells.

        :param biomarker_name: The name of the biomarker to analyze.
        :return: A tuple containing the mean and standard deviation of the biomarker expression levels.
        """
        biomarker_values = [cell.get_biomarker(biomarker_name) for cell in self.cells if cell.get_biomarker(biomarker_name) is not None]
        
        if not biomarker_values:
            raise ValueError(f"No data available for biomarker '{biomarker_name}'.")

        return np.mean(biomarker_values), np.std(biomarker_values)

    def to_graph(self, node_feature_fn, edge_index_fn, edge_attr_fn=None):
        """
        Converts the tissue into a graph structure compatible with PyTorch Geometric (PyG).

        :param node_feature_fn: A function that takes a Cell object and returns a list of features to be used as the node features.
        :param edge_index_fn: A function that takes a list of Cell objects and returns an edge index tensor.
        :param edge_attr_fn: An optional function that takes two Cell objects and returns the edge attribute for the edge between them.
        :return: A PyG Data object representing the tissue as a graph.
        """
        # Generate node features from each cell
        node_features = self._generate_node_features(node_feature_fn)

        # Generate edge index
        edge_index = edge_index_fn(self.cells)

        # Generate edge attributes (if provided)
        edge_attr = self._generate_edge_attributes(edge_attr_fn, edge_index)

        # Create a PyG Data object and return it
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    def _generate_node_features(self, node_feature_fn):
        """
        Generate the node features by applying the given node_feature_fn to each cell.
        This method optimizes the feature extraction by processing all cells in a single step.

        :param node_feature_fn: A function that generates node features from each cell.
        :return: A tensor of node features.
        """
        try:
            node_features = torch.tensor([node_feature_fn(cell) for cell in self.cells], dtype=torch.float)
            return node_features
        except Exception as e:
            raise ValueError(f"Error generating node features: {e}")

    def _generate_edge_attributes(self, edge_attr_fn, edge_index):
        """
        Generate edge attributes by applying the edge_attr_fn to each pair of cells specified by the edge_index.

        :param edge_attr_fn: A function that generates an edge attribute for each pair of cells.
        :param edge_index: The edge index that specifies the connections between nodes.
        :return: A tensor of edge attributes.
        """
        if edge_attr_fn is None:
            return None  # No edge attributes are needed

        edge_attr = []
        for edge in edge_index.t().tolist():  # Iterate over each edge (i, j)
            cell1 = self.get_cell_by_id(edge[0])
            cell2 = self.get_cell_by_id(edge[1])

            if cell1 and cell2:
                edge_attr.append(edge_attr_fn(cell1, cell2))
            else:
                raise ValueError(f"Error retrieving cells for edge: {edge}")

        return torch.tensor(edge_attr, dtype=torch.float)

    def __str__(self):
        """
        Return a string representation of the Tissue object, summarizing the tissue ID and the number of cells.
        """
        return f"Tissue {self.tissue_id} with {len(self.cells)} cells"

