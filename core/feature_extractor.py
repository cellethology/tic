from spacegm.embeddings_analysis import get_composition_vector


def extract_composition_generalized(subgraph, general_cell_types, cell_type_mapping):
    """
    Extract neighborhood composition grouped by general cell types.

    Args:
        subgraph (dict): Subgraph data.
        general_cell_types (dict): General cell type categories with their specific types.
        cell_type_mapping (dict): Mapping of specific cell type names to indices.

    Returns:
        dict: Composition data grouped by general cell types.
    """
    composition_vec = get_composition_vector(subgraph.get("subgraph", None), len(cell_type_mapping))
    generalized_composition = {}

    for general_type, specific_types in general_cell_types.items():
        indices = [cell_type_mapping[cell_type] for cell_type in specific_types if cell_type in cell_type_mapping]
        generalized_composition[general_type] = sum(composition_vec[idx] for idx in indices)

    return generalized_composition



