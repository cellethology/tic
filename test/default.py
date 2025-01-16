from spacegm.data import CellularGraphDataset

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