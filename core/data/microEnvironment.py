import numpy as np
import torch
from torch_geometric.data import Data
from collections import deque

class MicroE:
    """
    Class to represent the microenvironment of a center cell within a set of cells.

    This class builds a microenvironment using a k-hop strategy with a distance threshold and
    can convert the microenvironment to a graph representation (PyG Data object).
    """

    def __init__(self, center_cell, cells, k=3, r=200, distance_fn=None):
        """
        Initialize the MicroE object.

        :param center_cell: The central cell (Cell object).
        :param cells: List of all cells (Cell objects).
        :param k: The k-hop value, indicating the maximum number of hops (default 3).
        :param r: The distance radius for selecting neighbors (default 200).
        :param distance_fn: A custom distance function for calculating distance between cells (default Euclidean distance).
        """
        self.center_cell = center_cell
        self.cells = cells
        self.k = k
        self.r = r
        self.distance_fn = distance_fn if distance_fn else self.default_distance

        # Validate cells positions
        self.validate_cells_positions()

        # Build the microenvironment
        self.neighbors = self.build_microenvironment()

    def default_distance(self, cell1, cell2):
        """
        Default distance function: Euclidean distance.

        :param cell1: The first cell (Cell object).
        :param cell2: The second cell (Cell object).
        :return: Euclidean distance between the two cells.
        """
        return np.linalg.norm(np.array(cell1.pos) - np.array(cell2.pos))

    def validate_cells_positions(self):
        """
        Ensure that all cells in the list have the same number of dimensions in their position data.
        """
        if not self.cells:
            raise ValueError("Cell list is empty.")
        
        first_cell_pos_len = len(self.cells[0].pos)
        for cell in self.cells:
            if len(cell.pos) != first_cell_pos_len:
                raise ValueError(f"Inconsistent position dimensions for cell {cell.cell_id}. Expected {first_cell_pos_len}-dimensional positions.")

    def build_microenvironment(self):
        """
        Build the microenvironment by performing a BFS to find neighbors up to k-hops and within a distance threshold.
        
        :return: A list of neighboring cells.
        """
        visited = set()
        queue = deque([(self.center_cell, 0)])  # (cell, hop_level)
        visited.add(self.center_cell.cell_id)
        neighbors = []

        while queue:
            current_cell, hop_level = queue.popleft()

            if hop_level >= self.k:  # Stop if we've reached the k-hop limit
                continue

            # Check neighbors (cells within distance r)
            for neighbor_cell in self.cells:
                if neighbor_cell.cell_id == current_cell.cell_id:  # Skip the current cell
                    continue

                if neighbor_cell.cell_id not in visited:
                    distance = self.distance_fn(current_cell, neighbor_cell)

                    if distance <= self.r:  # Only include neighbors within the distance threshold
                        visited.add(neighbor_cell.cell_id)
                        queue.append((neighbor_cell, hop_level + 1))

                        if hop_level > 0:  # Add neighbors only after the 0-hop (center cell itself)
                            neighbors.append(neighbor_cell)

        return neighbors

    def get_neighbors(self):
        """
        Get the list of neighboring cells in the microenvironment.

        :return: A list of neighboring Cell objects.
        """
        return self.neighbors

    def to_graph(self, node_feature_fn, edge_index_fn, edge_attr_fn=None):
        """
        Convert the microenvironment into a graph structure compatible with PyTorch Geometric (PyG).

        :param node_feature_fn: A function that takes a Cell object and returns a list of features to be used as the node features.
        :param edge_index_fn: A function that takes a list of Cell objects and returns an edge index tensor.
        :param edge_attr_fn: An optional function that takes two Cell objects and returns the edge attribute for the edge between them.
        :return: A PyG Data object representing the microenvironment as a graph.
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

    def __str__(self):
        """
        Return a string representation of the MicroE object.
        """
        return f"Microenvironment around Cell {self.center_cell.cell_id} with {len(self.neighbors)} neighboring cells"
