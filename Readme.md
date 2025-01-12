# TIC: Temporal Inference of Cells
Temporal Inference of Cells (TIC) is a computational framework for analyzing cellular micro-environments using graph-based pseudo-time analysis. TIC integrates tools for graph construction, embedding preparation, pseudo-time trajectory computation, and biomarker trend visualization.

## Installation

To use TIC, you need to install its dependencies:

### Step 1: Clone the Repository
```
git clone https://github.com/JiahaoZhang-Public/tic.git
cd tic
git submodule update --init
```
### Step 2: Install Submodules
TIC relies on the following submodules:

* space-gm: For constructing graphs of cellular micro-environments.
* slingshot: Provides the core algorithm for pseudo-time trajectory computation.

Install these submodules:
```
# Install space-gm
git submodule update --init tools/space-gm
cd tools/space-gm
pip install -e .

# Install slingshot
cd ../slingshot
pip install -e .

# Return to the main directory
cd ../..
pip install -e .
```
## Project Framework
The directory structure for the TIC project is as follows:
```
TIC/
├── adapters/           # Adapter scripts for data and model integration
├── config/             # Configuration files (Hydra-based)
│   ├── pseudotime/     # Configuration files for pseudo-time analysis
├── core/               # Core modules for data processing and analysis
├── data/               # Directory for input datasets and intermediate files
├── notebook/           # Jupyter notebooks for example workflows
├── outputs/            # Output directory for pseudo-time analysis results
├── scripts/            # Scripts for running experiments and pipelines
├── tools/              # External tools and dependencies
├── utils/              # Utility scripts and helper functions
├── Readme.md           # Project documentation
├── requirements.txt    # Python dependencies
└── setup.py            # Installation script

```
### PseudoTime Analysis output dir
The results of the pseudo-time analysis are stored in a structured directory as follows:
```
outputs/
├── embedding_analysis/
│   ├── {embedding_key}/
│   │   ├── expression/
│   │   │   ├── biomarker_trends.png                    # Biomarker trends over pseudo-time
│   │   │   ├── cluster_biomarker_summary.csv           # Aggregated biomarker data
│   │   │   └── cluster_summary.csv                     # Clustering summary
│   │   └── pseudotime/
│   │       ├── start_node_{n}/                         # Pseudo-time results for start node {n}
│   │       │   ├── Biomarkers_trends_vs_pseudotime.png # Biomarker trends over pseudo-time
│   │       │   ├── Neighborhood_Composition_trends_vs_pseudotime.png # Neighborhood composition trends
│   │       │   ├── pseudotime.csv                      # Computed pseudo-time values
│   │       └── pseudotime_visualization.png            # Visualization of pseudo-time trajectories
│   ├── umap/
│   │   ├── umap_vs_cell_types.png                      # UMAP embeddings colored by cell types

```