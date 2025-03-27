# core.data.cell

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

    def __init__(self,tissue_id, cell_id, pos, size, cell_type=None, biomarkers=None, **additional_features):
        """
        Initializes the Cell object with the provided attributes.
        
        :param tissue_id: Unique identifier for the tissue or region.
        :param cell_id: Unique identifier for the cell.
        :param pos: The cell's spatial position (x, y, z).
        :param size: The cell's size or volume.
        :param cell_type: The type of the cell (e.g., "Tumor", "T cell").
        :param biomarkers: A Biomarkers object containing the cell's biomarker data (default is empty).
        :param additional_features: Additional features of the cell (e.g., gene expression, protein levels).
        """
        self.tissue_id = tissue_id
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
