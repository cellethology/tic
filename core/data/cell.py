from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import os
import pickle
import pandas as pd
import numpy as np
import torch

class AbstractCellInfoDataset(ABC):
    """
    Abstract base class for a dataset containing cell information. This class defines the common interface
    for working with different datasets that use the `CellInfo` format.
    """
    
    @abstractmethod
    def load_raw_data(self, region_id: str, cell_id: int, raw_dir: str) -> Dict[str, Any]:
        """
        Loads the raw data (coordinates, biomarker expression, size, etc.) for a specific cell in a given region.
        
        Args:
            region_id (str): The region ID for the cell.
            cell_id (int): The cell ID.
            raw_dir (str): The directory containing the raw data files.
        
        Returns:
            Dict: A dictionary containing the cell's data (coordinates, size, biomarker expression, etc.).
        """
        pass
    
    @abstractmethod
    def create_cell_info(self, region_id: str, cell_id: int, raw_dir: str, embedding: Optional[np.ndarray] = None) -> "CellInfo":
        """
        Initializes and returns a `CellInfo` object for a given region and cell ID.
        
        Args:
            region_id (str): The region ID for the cell.
            cell_id (int): The cell ID.
            raw_dir (str): The directory containing the raw data files.
            embedding (Optional[np.ndarray]): The embedding of the cell (if available).
        
        Returns:
            CellInfo: The initialized `CellInfo` object for the given region and cell.
        """
        pass

    @abstractmethod
    def save_cell_info(self, cell_info: "CellInfo", filepath: str):
        """
        Saves the `CellInfo` object to a file.
        
        Args:
            cell_info (CellInfo): The `CellInfo` object to save.
            filepath (str): The path where the object will be saved.
        """
        pass
    
    @abstractmethod
    def load_cell_info(self, filepath: str) -> "CellInfo":
        """
        Loads a `CellInfo` object from a file.
        
        Args:
            filepath (str): The path of the file containing the serialized `CellInfo` object.
        
        Returns:
            CellInfo: The loaded `CellInfo` object.
        """
        pass

class CellInfo:
    """
    A class to represent information about a cell in a tissue region.
    
    Attributes:
        cell_id (int): The ID of the cell.
        region_id (str): The ID of the region the cell belongs to.
        coordinates (List[float]): The X, Y (and Z) coordinates of the cell.
        size (float): The size of the cell.
        biomarker_expression (Dict[str, float]): Dictionary containing biomarker expression values.
        embedding (np.ndarray or torch.Tensor): The embedding of the cell (obtained after GNN processing).
        pseudotime (float): The pseudotime of the cell.
    """

    def __init__(self, 
                 region_id: str, 
                 cell_id: int,
                 raw_dir: str,
                 embedding: np.ndarray or torch.Tensor = None,  # type: ignore
                 pseudotime: float = None):
        self.cell_id = cell_id
        self.region_id = region_id
        self.coordinates = None
        self.size = None
        self.biomarker_expression = {}
        self.embedding = embedding
        self.umap_embedding = None  # 2D UMAP embedding
        self.cluster_label = None  # Cluster label assigned to the cell
        self.pseudotime = pseudotime
        
        # Initialize the cell data by reading from raw files in the specified directory
        self._initialize_from_raw_data(raw_dir)
    
    def __init__(self, pkl_path: str):
        """
        Initialize the CellInfo object by loading it from a serialized pickle file.
        
        Args:
            pkl_path (str): The path to the pickle file containing the serialized CellInfo object.
        """
        self._load_from_pickle(pkl_path)

    def _initialize_from_raw_data(self, raw_dir: str):
        """
        Initializes the attributes of the cell from the raw data located in the specified directory.
        
        Args:
            raw_dir (str): The directory containing raw data files for the region.
        """
        # Paths for raw data files
        paths = self._get_raw_data_paths(raw_dir)

        # Read data from files and initialize attributes
        cell_data, expression, cell_features, cell_types = self._read_raw_data(paths)
        
        self.coordinates = self._get_coordinates(cell_data)
        self.size = self._get_size(cell_features)
        self.biomarker_expression = self._get_biomarker_expression(expression)
    
    def _get_raw_data_paths(self, raw_dir: str):
        """
        Helper function to generate file paths for the raw data.
        
        Args:
            raw_dir (str): Directory containing the raw data files.
            
        Returns:
            dict: Dictionary with paths to raw data files.
        """
        return {
            "cell_data": os.path.join(raw_dir, f"{self.region_id}.cell_data.csv"),
            "expression": os.path.join(raw_dir, f"{self.region_id}.expression.csv"),
            "cell_features": os.path.join(raw_dir, f"{self.region_id}.cell_features.csv"),
            "cell_types": os.path.join(raw_dir, f"{self.region_id}.cell_types.csv")
        }
    
    def _read_raw_data(self, paths: dict):
        """
        Helper function to read raw data from CSV files.
        
        Args:
            paths (dict): Dictionary with file paths for the raw data files.
        
        Returns:
            tuple: Tuple containing data from the cell, expression, features, and types CSVs.
        """
        try:
            cell_data = pd.read_csv(paths["cell_data"], index_col="CELL_ID")
            expression = pd.read_csv(paths["expression"], index_col="CELL_ID")
            cell_features = pd.read_csv(paths["cell_features"], index_col="CELL_ID")
            cell_types = pd.read_csv(paths["cell_types"], index_col="CELL_ID")
        except FileNotFoundError as e:
            print(f"Error loading data from {e.filename}. Please check the file paths.")
            raise
        return cell_data, expression, cell_features, cell_types

    def _get_coordinates(self, cell_data: pd.DataFrame):
        """
        Extract the coordinates (X, Y) of the cell from the cell data.
        
        Args:
            cell_data (pd.DataFrame): DataFrame containing cell data.
        
        Returns:
            list: Coordinates of the cell (X, Y).
        """
        return [
            cell_data.loc[self.cell_id, "X"] if self.cell_id in cell_data.index else None,
            cell_data.loc[self.cell_id, "Y"] if self.cell_id in cell_data.index else None
        ]
    
    def _get_size(self, cell_features: pd.DataFrame):
        """
        Extract the size of the cell from the cell features.
        
        Args:
            cell_features (pd.DataFrame): DataFrame containing cell features.
        
        Returns:
            float: The size of the cell.
        """
        return cell_features.loc[self.cell_id, "SIZE"] if self.cell_id in cell_features.index else None
    
    def _get_biomarker_expression(self, expression: pd.DataFrame):
        """
        Extract the biomarker expression data for the cell.
        
        Args:
            expression (pd.DataFrame): DataFrame containing biomarker expression data.
        
        Returns:
            dict: Dictionary of biomarker expressions for the cell.
        """
        biomarker_expression = {}
        for biomarker in expression.columns:
            if biomarker != "ACQUISITION_ID" and self.cell_id in expression.index:
                biomarker_expression[biomarker] = expression.loc[self.cell_id, biomarker]
            else:
                biomarker_expression[biomarker] = None
        return biomarker_expression

    def _load_from_pickle(self, pkl_path: str):
        """
        Load the attributes of the cell from a serialized pickle file.
        
        Args:
            pkl_path (str): The path to the pickle file.
        """
        try:
            with open(pkl_path, 'rb') as f:
                cell_info = pickle.load(f)
            self.cell_id = cell_info.cell_id
            self.region_id = cell_info.region_id
            self.coordinates = cell_info.coordinates
            self.size = cell_info.size
            self.biomarker_expression = cell_info.biomarker_expression
            self.embedding = cell_info.embedding
            self.umap_embedding = cell_info.umap_embedding
            self.cluster_label = cell_info.cluster_label
            self.pseudotime = cell_info.pseudotime
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print(f"Error loading pickle file {pkl_path}: {e}")
            raise
    
    def update_biomarker_expression(self, biomarker_expression: Dict[str, float]):
        """
        Update the biomarker expression values for the cell.
        
        Args:
            biomarker_expression (Dict[str, float]): Dictionary containing updated biomarker expression values.
        """
        self.biomarker_expression.update(biomarker_expression)

    def set_embedding(self, embedding: np.ndarray or torch.Tensor):  # type: ignore
        """
        Set the embedding of the cell.
        
        Args:
            embedding (np.ndarray or torch.Tensor): The embedding of the cell.
        """
        self.embedding = embedding
    
    def set_umap_embedding(self, umap_embedding: np.ndarray):
        """
        Set the 2D UMAP embedding of the cell.
        
        Args:
            umap_embedding (np.ndarray): The 2D UMAP embedding of the cell.
        """
        self.umap_embedding = umap_embedding

    def set_cluster_label(self, cluster_label: int):
        """
        Set the cluster label for the cell.
        
        Args:
            cluster_label (int): The cluster label assigned to the cell.
        """
        self.cluster_label = cluster_label
    
    def set_pseudotime(self, pseudotime: float):
        """
        Set the pseudotime for the cell.
        
        Args:
            pseudotime (float): The pseudotime value for the cell.
        """
        self.pseudotime = pseudotime
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the cell's data to a dictionary format for easier analysis or exporting.
        
        Returns:
            Dict[str, Any]: A dictionary containing the cell's data.
        """
        return {
            "cell_id": self.cell_id,
            "region_id": self.region_id,
            "coordinates": self.coordinates,
            "size": self.size,
            "biomarker_expression": self.biomarker_expression,
            "embedding": self.embedding.tolist() if isinstance(self.embedding, torch.Tensor) else self.embedding,
            "umap_embedding": self.umap_embedding.tolist() if self.umap_embedding is not None else None,
            "cluster_label": self.cluster_label,
            "pseudotime": self.pseudotime
        }
    
    def __repr__(self):
        """
        Custom string representation of the object to print cell info.
        """
        return f"CellInfo(cell_id={self.cell_id}, region_id={self.region_id}, pseudotime={self.pseudotime})"
    
    def save(self, filepath: str):
        """
        Save the CellInfo object to a file using pickle.
        
        Args:
            filepath (str): The path where the object will be saved.
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
            print(f"CellInfo object saved to {filepath}")
        except IOError as e:
            print(f"Error saving object to {filepath}: {e}")
    
    @staticmethod
    def load(filepath: str):
        """
        Load a CellInfo object from a file.
        
        Args:
            filepath (str): The path of the file containing the serialized object.
        
        Returns:
            CellInfo: The loaded CellInfo object.
        """
        try:
            with open(filepath, 'rb') as f:
                cell_info = pickle.load(f)
            print(f"CellInfo object loaded from {filepath}")
            return cell_info
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print(f"Error loading object from {filepath}: {e}")
            raise


class CellBatch:
    """
    A class to store a batch of CellInfo instances.
    Supports loading from files and processing for downstream tasks.
    """    
    def __init__(self, cell_infos: List[CellInfo] = None, pkl_dir: str = None):
        """
        Initialize the CellBatch either by using pre-existing CellInfo instances
        or by loading serialized CellInfo objects from pickle files.
        
        Args:
            cell_infos (List[CellInfo], optional): List of CellInfo instances.
            pkl_dir (str, optional): Directory containing serialized CellInfo objects.
        """
        self.cells = cell_infos if cell_infos else []
        
        if pkl_dir:
            self.load_from_folder(pkl_dir)
    
    def load_from_folder(self, pkl_dir: str):
        """
        Load serialized CellInfo objects from pickle files in the specified folder.
        
        Args:
            pkl_dir (str): The directory containing serialized `.pkl` files.
        """
        try:
            for file_name in os.listdir(pkl_dir):
                if file_name.endswith(".pkl"):
                    file_path = os.path.join(pkl_dir, file_name)
                    self.cells.append(CellInfo(file_path))
        except FileNotFoundError as e:
            print(f"Error loading folder {pkl_dir}: {e}")
    
    def add_embeddings(self, embeddings: np.ndarray):
        """Add embeddings to each cell in the batch."""
        for cell, embedding in zip(self.cells, embeddings):
            cell.set_embedding(embedding)
    
    def add_cluster_labels(self, cluster_labels: np.ndarray):
        """Add cluster labels to each cell in the batch."""
        for cell, cluster_label in zip(self.cells, cluster_labels):
            cell.set_cluster_label(cluster_label)
    
    def add_umap_embeddings(self, umap_embeddings: np.ndarray):
        """Add UMAP embeddings to each cell in the batch."""
        for cell, umap_embedding in zip(self.cells, umap_embeddings):
            cell.set_umap_embedding(umap_embedding)
    
    def add_pseudotimes(self, pseudotimes: np.ndarray):
        """Add pseudotimes to each cell in the batch."""
        for cell, pseudotime in zip(self.cells, pseudotimes):
            cell.set_pseudotime(pseudotime)

    def get_embeddings(self) -> np.ndarray:
        """
        Get the embeddings of all cells in the batch as a numpy array.
        
        Returns:
            np.ndarray: The embeddings of all cells in the batch.
        """
        embeddings = [cell.embedding if cell.embedding is not None else np.zeros_like(self.cells[0].embedding) 
                      for cell in self.cells]
        return np.array(embeddings)

    def get_umap_embeddings(self) -> np.ndarray:
        """
        Get the UMAP embeddings of all cells in the batch as a numpy array.
        
        Returns:
            np.ndarray: The UMAP embeddings of all cells in the batch.
        """
        umap_embeddings = [cell.umap_embedding if cell.umap_embedding is not None else np.zeros_like(self.cells[0].umap_embedding)
                           for cell in self.cells]
        return np.array(umap_embeddings)
    
    def get_cluster_labels(self) -> np.ndarray:
        """
        Get the cluster labels of all cells in the batch as a numpy array.
        
        Returns:
            np.ndarray: The cluster labels of all cells in the batch.
        """
        return np.array([cell.cluster_label for cell in self.cells])
    
    def get_pseudotimes(self) -> np.ndarray:
        """
        Get the pseudotimes of all cells in the batch as a numpy array.
        
        Returns:
            np.ndarray: The pseudotimes of all cells in the batch.
        """
        return np.array([cell.pseudotime for cell in self.cells])
    
    def get_biomarker_expression(self):
        """Get the biomarker expression values of all cells in the batch."""
        return [cell.biomarker_expression for cell in self.cells]
    
    def to_dict(self):
        """Convert the batch's data to a dictionary format."""
        return [cell.to_dict() for cell in self.cells]
    
    def save(self, output_dir: str):
        """Save all CellInfo objects in the batch."""
        os.makedirs(output_dir, exist_ok=True)
        for cell in self.cells:
            filepath = os.path.join(output_dir, f"{cell.region_id}_{cell.cell_id}.pkl")
            cell.save(filepath)
    
    def __repr__(self):
        """Custom string representation of the object."""
        return f"CellBatch(num_cells={len(self.cells)})"

#---------------------------
# New Representation of Cell
#---------------------------

class Biomarkers:
    """
    A class to manage biomarkers and their expression levels.

    This class allows you to store biomarker data and retrieve the expression level of each biomarker.
    Biomarkers are stored as a dictionary, where the keys are biomarker names and the values are their expression levels.
    """

    def __init__(self, **biomarker_values):
        """
        Initializes the Biomarkers object with dynamic biomarker data.
        
        :param biomarker_values: Keyword arguments representing biomarker names and their corresponding expression levels.
        """
        self.biomarkers = biomarker_values
    
    def __getattr__(self, biomarker_name):
        """
        Retrieve the expression level of a specified biomarker.

        :param biomarker_name: The name of the biomarker to retrieve.
        :return: The expression level of the biomarker if it exists, else raises AttributeError.
        """
        biomarker_dict = self.__dict__.get("biomarkers", {})
        if biomarker_name in biomarker_dict:
            return biomarker_dict[biomarker_name]
        else:
            raise AttributeError(f"Biomarker '{biomarker_name}' not found in this cell.")

    def __repr__(self):
        """ 
        Returns a string representation of the biomarkers and their expression levels.
        This is useful for debugging and logging.
        """
        return f"Biomarkers({self.biomarkers})"


class Cell:
    """
    A class representing a single cell with its attributes, including position, size, biomarkers, and other features.

    This class stores information about the cell's ID, position, size, type, biomarkers, and any additional features.
    It allows for easy retrieval of biomarker information and additional features.
    """

    def __init__(self, cell_id, pos, size, cell_type=None, biomarkers=None, **additional_features):
        """
        Initializes the Cell object with the provided attributes.
        
        :param cell_id: Unique identifier for the cell.
        :param pos: The cell's spatial position (x, y, z).
        :param size: The cell's size or volume.
        :param cell_type: The type of the cell (e.g., "Tumor", "T cell").
        :param biomarkers: A Biomarkers object containing the cell's biomarker data (default is empty).
        :param additional_features: Additional features of the cell (e.g., gene expression, protein levels).
        """
        self.cell_id = cell_id
        self.pos = pos
        self.size = size
        self.cell_type = cell_type
        self.biomarkers = biomarkers if biomarkers else Biomarkers()  # Default to empty biomarkers
        self.additional_features = additional_features
    
    def __str__(self):
        """
        Provides a string representation of the Cell object, including basic information such as its ID, position, and size.
        
        :return: A string describing the cell.
        """
        return f"Cell {self.cell_id} at position {self.pos} with size {self.size} and type {self.cell_type}"
    
    def get_biomarker(self, biomarker_name):
        """
        Retrieves the expression level of a specific biomarker.
        
        :param biomarker_name: The name of the biomarker to retrieve.
        :return: The expression level of the biomarker if it exists, else None.
        """
        try:
            return self.biomarkers.__getattr__(biomarker_name)
        except AttributeError:
            print(f"Warning: Biomarker '{biomarker_name}' not found in cell {self.cell_id}.")
            return None  # Return None if biomarker doesn't exist
    
    def add_feature(self, feature_name, feature_value):
        """
        Adds or updates an additional feature for the cell.
        
        :param feature_name: The name of the feature.
        :param feature_value: The value of the feature (e.g., gene expression level).
        """
        self.additional_features[feature_name] = feature_value
    
    def get_feature(self, feature_name):
        """
        Retrieves the value of a specific additional feature.
        
        :param feature_name: The name of the feature to retrieve.
        :return: The value of the feature if it exists, else None.
        """
        return self.additional_features.get(feature_name, None)
