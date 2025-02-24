import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from spacegm.utils import CELL_TYPE_MAPPING_UPMC

def plot_subgraph(subgraph_data, cell_type_mapping=CELL_TYPE_MAPPING_UPMC):
    """
    Plots the k-hop subgraph with the center node highlighted and labeled with its cell type.
    
    Args:
        subgraph_data (Data): The PyG Data object containing the subgraph.
        cell_type_mapping (dict): A mapping of cell type indices to names.
    """
    # Create a NetworkX graph from the PyG data
    G = nx.Graph()
    
    # Add nodes with positions and features (cell_type and biomarker expression)
    for i, (node_features, pos) in enumerate(zip(subgraph_data.x, subgraph_data.pos)):
        G.add_node(i, pos=pos, features=node_features)
    
    # Add edges between nodes
    for edge in subgraph_data.edge_index.t().tolist():
        G.add_edge(edge[0], edge[1])
    
    # Get the positions for the plot
    pos = {i: subgraph_data.pos[i].numpy() for i in range(len(subgraph_data.pos))}
    
    # Get the cell type of the center node
    center_node_idx = subgraph_data.relabeled_node_idx.numpy()[0]
    center_node_features = subgraph_data.x[center_node_idx].numpy()
    cell_type_idx = np.argmax(center_node_features[-len(cell_type_mapping):])  # Get index of the cell type (one-hot encoding)
    cell_type = list(cell_type_mapping.keys())[cell_type_idx]

    # Create a color map for the plot (highlight center node)
    node_colors = ['red' if i == center_node_idx else 'lightblue' for i in range(len(G.nodes))]

    # Plot the subgraph
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color=node_colors, font_size=12, font_weight='bold')
    
    # Add label for the center node
    plt.text(
        pos[center_node_idx][0], pos[center_node_idx][1], 
        f'Center\n({cell_type})', 
        horizontalalignment='center', fontsize=10, color='black', weight='bold'
    )
    
    plt.title(f'k-hop Subgraph (Center Node: {cell_type})')
    plt.show()