# Pseudo-Time Analysis Pipeline Configuration Guide

This document provides guidance on how to customize and modify the configuration files for the pseudo-time analysis pipeline using Hydra.

## Configuration Overview

The pipeline configuration is managed by Hydra, a powerful configuration framework. The configuration files are stored in YAML format and are located in the `config/pseudotime/` directory.

### Main Configuration Components

1. **General Settings (`general`)**
   - Defines the root directories, device settings, and global parameters for the pipeline.

2. **Dataset Configuration (`dataset`)**
   - Contains dataset-specific parameters like cell type mappings, biomarkers, and processing settings.

3. **Sampler Configuration (`sampler`)**
   - Specifies the settings for subgraph sampling, including the number of samples to generate and the regions/cell types to sample from.

4. **Embedding Configuration (`embedding`)**
   - Contains settings related to embedding methods, clustering methods, and the number of clusters to use.

5. **Pseudo-Time Configuration (`pseudotime`)**
   - Defines the start nodes for the pseudo-time analysis.

6. **Causal Inference Configuration (`causal`)**
   - Provides settings for causal inference, such as the choice of causal inference method and the biomarkers for analysis.

7. **Constants (`constants`)**
   - Stores static mappings and lists like cell type mappings, biomarker lists, and cell type frequencies.

---

## How to Modify Configurations:
You just need to modify: general,sampler,embedding,pseudotime,causal;Do not change dataset or constants unless you are confident what you have changed

### 1. General Settings

The `general` section defines paths and global parameters for the pipeline.

Example:
```yaml
general:
  raw_dir: "/path/to/your/data"  # Directory containing raw data (e.g., cell_data.csv, expression.csv)
  output_dir: "/path/to/your/output"  # Directory to save pipeline outputs
  data_root: "/path/to/your/data"  # Root directory for your CellularGraphDataset (refer to the space-gm tutorial)
  random_seed: 42  # Random seed for reproducibility
  show_plots: True  # Whether to display plots during execution
```
### 2. Dataset Configuration
The `dataset` section contains parameters for dataset processing.
```yaml
dataset:
  raw_folder_name: "graph"  # Raw folder name for the graph data
  processed_folder_name: "tg_graph"  # Processed graph folder name
  node_features:  # Features for each node (cell)
    - "cell_type"
    - "SIZE"
    - "biomarker_expression"
    - "neighborhood_composition"
    - "center_coord"
  edge_features:  # Features for each edge (e.g., relationships between cells)
    - "edge_type"
    - "distance"
  biomarkers: ${constants.BIOMARKERS_UPMC}  # List of biomarkers (from constants)
  subgraph_size: 3  # Size of the subgraph used in graph construction
```
### 3. Sampler Configuration
The `sampler` section controls subgraph sampling parameters.
```yaml
sampler:
  total_samples: 1000  # Total number of samples to generate
  regions: 'all'  # Regions to sample from (e.g., ['region1', 'region2'] or 'all')
  cell_types: ['Tumor']  # Cell types to sample from (e.g., ['cell_type1', 'cell_type2'])
```
### 4. Embedding Configuration
The `embedding` section defines the methods and settings for generating cell embeddings.

Example:
```yaml
embedding:
  methods: ['composition_vectors', 'expression_vectors']  # Embedding methods to use
  n_pca_components: 10  # Number of PCA components to retain
  cluster_method: 'kmeans'  # Clustering method (e.g., 'kmeans', 'dbscan')
  n_clusters: 2  # Number of clusters to use for clustering
```
### 5. Pseudo-Time Configuration
The `pseudotime` section specifies the start nodes for the pseudo-time analysis.

Example:
```yaml
pseudotime:
  start_nodes: [0, 1]  # List of start nodes (clusters) for pseudo-time analysis
```
### 6. Causal Inference Configuration
The `causal` section provides parameters for performing causal inference.

Example:
```yaml
causal:
  pseudotime_file: "/path/to/pseudotime.csv"  # Path to the pseudotime file
  method: 'GrangerCausality'  # Causal inference method ('GrangerCausality' or 'LinearRegressionWithControls')
  included_cell_types:  # Cell types to include in the causal analysis
    - Tumor
    - Naive immune cell
    - CD4 T cell
    - CD8 T cell
  included_biomarkers:  # Biomarkers to include in the causal analysis
    - PanCK
    - aSMA
    - CD45
    - Vimentin
  sparsity_threshold: 0.1  # Filter threshold for sparsity in biomarker data
  target_biomarker: 'PanCK'  # Target biomarker for causal inference
  visualization:
    top_results: 10  # Number of top results to visualize by P-Value
    output_dir: ${general.output_dir}/visualization  # Output directory for visualizations
```
### 7. Constants
The `constants` section contains mappings and predefined lists.