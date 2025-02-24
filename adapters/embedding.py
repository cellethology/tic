import numpy as np
from spacegm.data import CellularGraphDataset
from spacegm.embeddings_analysis import get_composition_vector

from adapters.sampler import CustomSubgraphSampler, get_subgraph_by_cell
from core.pseudotime_analysis import CellEmbedding

def initialize_dataset():
    dataset_root = "data/example"
    dataset_kwargs = {
        'transform': [],
        'pre_transform': None,
        'raw_folder_name': 'graph',  # os.path.join(dataset_root, "graph") is the folder where we saved nx graphs
        'processed_folder_name': 'tg_graph',  # processed dataset files will be stored here
        'node_features': ["cell_type", "SIZE", "biomarker_expression", "neighborhood_composition", "center_coord"], 
        'edge_features': ["edge_type", "distance"],

        'subgraph_size': 3,  # indicating we want to sample 3-hop subgraphs from these regions (for training/inference), this is a core parameter for SPACE-GM.
        'subgraph_source': 'on-the-fly',
        'subgraph_allow_distant_edge': True,
        'subgraph_radius_limit': 200.,
    }

    feature_kwargs = {
        "biomarker_expression_process_method": "linear",
        "biomarker_expression_lower_bound": 0,
        "biomarker_expression_upper_bound": 18,
        "neighborhood_size": 10,
    }
    dataset_kwargs.update(feature_kwargs)

    dataset = CellularGraphDataset(dataset_root, **dataset_kwargs)
    return dataset

def Config():
    config = {
        "embedding_preparation": {
            "keys": ["composition_vectors", "expression_vectors"]
        }
    }
    return config
    
def prepare_embeddings(dataset, sampled_cells, config):
    """Prepare embeddings based on sampled cells and add them to the embedding input structure."""
    embeddings_dict = {'composition_vectors': [], 'expression_vectors': []}
    identifiers = []

    for region_id, cell_id in sampled_cells:
        subgraph = get_subgraph_by_cell(dataset, region_id, cell_id)
        
        # Composition vectors
        if "composition_vectors" in config.methods:
            composition_vector = get_composition_vector(subgraph, n_cell_types=len(dataset.cell_type_mapping))
            embeddings_dict['composition_vectors'].append(composition_vector)
        
        # Expression vectors
        if "expression_vectors" in config.methods:
            expression_vector = extract_expression_vector(subgraph,node_feature_names=dataset.node_feature_names)
            embeddings_dict['expression_vectors'].append(expression_vector)
        
        identifiers.append([region_id, cell_id])

    # Handle concatenated embeddings if specified
    concatenated_keys = [key for key in config.methods if "+" in key]
    for key in concatenated_keys:
        components = key.split("+")
        concatenated_embeddings = [
            np.concatenate([embeddings_dict[comp][i] for comp in components if comp in embeddings_dict], axis=None)
            for i in range(len(sampled_cells))
        ]
        embeddings_dict[key] = concatenated_embeddings

    return CellEmbedding(identifiers=identifiers, embeddings=embeddings_dict)

def extract_expression_vector(subgraph, node_feature_names):
    """
    Extract expression vector for the central node in the subgraph based on a list of node feature names.
    
    Args:
        subgraph (torch_geometric.data.Data): The subgraph object with node features.
        node_feature_names (list): List of all node feature names as they appear in the subgraph node features.

    Returns:
        numpy.ndarray: Array of biomarker expression values for the central node.
    """
    if not hasattr(subgraph, 'x') or subgraph.x is None:
        raise ValueError("Subgraph does not contain node features ('x') or is incorrectly formatted.")
    
    # Get the indices for biomarker expressions in the feature matrix
    biomarker_indices = [i for i, name in enumerate(node_feature_names) if name.startswith('biomarker_expression-')]

    # Assuming central node is at index 0, you can adjust if different
    central_node_features = subgraph.x[0]  # Modify if the central node index is stored differently
    
    # Extract the biomarker expressions using the calculated indices
    expression_vector = central_node_features[biomarker_indices]
    return expression_vector.numpy()


if __name__ == "__main__":
    dataset = initialize_dataset()
    print(dataset.node_feature_names)
    config = Config()
    sampler = CustomSubgraphSampler(raw_dir="data/example/voronoi")
    sampled_cells = sampler.sample(total_samples=10)
    prepared_embeddings = prepare_embeddings(dataset,sampled_cells,config)